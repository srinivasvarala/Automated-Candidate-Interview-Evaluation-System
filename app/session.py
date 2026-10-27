"""Interview session module.

One Interview session is one run from setup to Scorecard. The module owns
input validation, driving the interview graph, the Transcript, and saving
after every Turn. Callers (the WebSocket route, tests) see only `start`,
`answer` and `end`, each an async iterator of typed events.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app import database
from app.agents import tier_name
from app.config import settings

logger = logging.getLogger(__name__)

INTERVIEW_TYPES = ("mixed", "behavioral", "technical", "system_design")
END_SENTINEL = "__END_INTERVIEW__"
MAX_JOB_POSITION = 200
MAX_JOB_DESCRIPTION = 10_000
MAX_RESUME = 10_000


# --- Events ---

@dataclass(frozen=True)
class Started:
    content: str
    num_questions: int


@dataclass(frozen=True)
class Typing:
    pass


@dataclass(frozen=True)
class AgentMessage:
    source: str  # "interviewer" | "evaluator"
    content: str
    score: int | None = None  # evaluator only
    tier: str | None = None
    tier_index: int | None = None


@dataclass(frozen=True)
class WaitingForInput:
    pass


@dataclass(frozen=True)
class Summary:
    content: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Complete:
    pass


@dataclass(frozen=True)
class Error:
    content: str
    fatal: bool = False  # fatal: the session cannot continue


Event = Started | Typing | AgentMessage | WaitingForInput | Summary | Complete | Error


class InterviewSession:
    def __init__(self, graph, store=database):
        self._graph = graph
        self._store = store
        self.id = str(uuid.uuid4())
        self.status = "new"  # new | in_progress | completed | failed
        self.questions_asked = 0
        self.transcript: list[dict] = []
        self._saved = 0  # transcript messages already persisted
        self._awaiting = False
        self._summary_data: dict | None = None
        self._config = {"configurable": {"thread_id": self.id}}

    @property
    def awaiting_input(self) -> bool:
        return self._awaiting

    # --- Interface ---

    async def start(
        self,
        job_position: str,
        job_description: str = "",
        resume_text: str = "",
        interview_type: str = "mixed",
    ) -> AsyncIterator[Event]:
        if self.status != "new":
            yield Error("Interview already started.")
            return

        job_position = (job_position or "").strip()
        if not job_position:
            yield Error("Job position is required.", fatal=True)
            return
        if len(job_position) > MAX_JOB_POSITION:
            yield Error(f"Job position must be under {MAX_JOB_POSITION} characters.", fatal=True)
            return
        job_description = (job_description or "").strip()
        if len(job_description) > MAX_JOB_DESCRIPTION:
            yield Error("Job description must be under 10,000 characters.", fatal=True)
            return
        resume_text = (resume_text or "").strip()[:MAX_RESUME]
        interview_type = (interview_type or "mixed").strip()
        if interview_type not in INTERVIEW_TYPES:
            interview_type = "mixed"

        logger.info(
            "[%s] Starting interview: position=%s, type=%s, has_resume=%s",
            self.id, job_position, interview_type, bool(resume_text),
        )

        try:
            await self._store.create_interview(self.id, job_position, job_description, interview_type)
        except Exception:
            logger.error("[%s] Failed to create interview record", self.id, exc_info=True)
            yield Error("Could not start the interview. Please try again.", fatal=True)
            return
        self.status = "in_progress"

        yield Started(
            content=f"Starting {interview_type} interview for {job_position}...",
            num_questions=settings.num_questions,
        )
        initial_state = {
            "messages": [HumanMessage(content=f"Start the interview for a {job_position} position.")],
            "job_position": job_position,
            "job_description": job_description,
            "resume_text": resume_text,
            "interview_type": interview_type,
            "questions_asked": 0,
            "current_phase": "interviewer",
            "end_requested": False,
            "summary_data": None,
            "performance_tracker": None,
        }
        async for event in self._run_turn(initial_state):
            yield event

    async def answer(self, text: str) -> AsyncIterator[Event]:
        if not self._awaiting:
            yield Error("The interview is not waiting for an answer.")
            return
        text = (text or "").strip()
        if len(text) > settings.max_answer_length:
            yield Error(f"Answer too long. Maximum {settings.max_answer_length} characters.")
            return
        if not text:
            yield Error("Please provide an answer.")
            return
        logger.info("[%s] Got user answer (%d chars)", self.id, len(text))
        self.questions_asked += 1
        async for event in self._run_turn(Command(resume=text)):
            yield event

    async def end(self) -> AsyncIterator[Event]:
        if not self._awaiting:
            yield Error("The interview is not waiting for an answer.")
            return
        logger.info("[%s] User requested end of interview", self.id)
        async for event in self._run_turn(Command(resume=END_SENTINEL)):
            yield event

    # --- Internals ---

    async def _run_turn(self, graph_input) -> AsyncIterator[Event]:
        """Run the graph until it pauses for input or finishes; save the result."""
        self._awaiting = False
        yield Typing()
        hit_interrupt = False
        try:
            stream = self._graph.astream(graph_input, config=self._config, stream_mode="updates")
            async for chunk in stream:
                for node_name, updates in chunk.items():
                    if node_name == "__interrupt__":
                        hit_interrupt = True
                    elif isinstance(updates, dict):
                        for event in self._events_from_update(updates):
                            yield event
            await self._save_progress()
            if not hit_interrupt:
                await self._store.finish_interview(self.id, self.questions_asked, self._summary_data)
        except Exception:
            logger.error("[%s] Turn failed", self.id, exc_info=True)
            self.status = "failed"
            yield Error("Something went wrong. Please start a new interview.", fatal=True)
            return

        if hit_interrupt:
            self._awaiting = True
            yield WaitingForInput()
        else:
            self.status = "completed"
            logger.info("[%s] Interview complete", self.id)
            yield Complete()

    def _events_from_update(self, updates: dict) -> list[Event]:
        events: list[Event] = []
        for msg in updates.get("messages", []):
            if isinstance(msg, AIMessage) and msg.name:
                self.transcript.append({"role": "assistant", "name": msg.name, "content": msg.content})
                if msg.name == "summary":
                    self._summary_data = updates.get("summary_data")
                    events.append(Summary(msg.content, self._summary_data or {}))
                else:
                    events.append(self._agent_message(msg, updates))
            elif isinstance(msg, HumanMessage):
                self.transcript.append({"role": "user", "name": getattr(msg, "name", ""), "content": msg.content})
        return events

    @staticmethod
    def _agent_message(msg: AIMessage, updates: dict) -> AgentMessage:
        if msg.name == "evaluator":
            tracker = updates.get("performance_tracker")
            if tracker and tracker.get("scores"):
                tier_index = tracker["tiers"][-1] if tracker.get("tiers") else 0
                return AgentMessage(
                    source=msg.name,
                    content=msg.content,
                    score=tracker["scores"][-1],
                    tier=tier_name(tier_index),
                    tier_index=tier_index,
                )
        return AgentMessage(source=msg.name, content=msg.content)

    async def _save_progress(self) -> None:
        new = self.transcript[self._saved:]
        await self._store.append_messages(self.id, new, self.questions_asked)
        self._saved = len(self.transcript)
