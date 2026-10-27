"""Interview session module.

One Interview session is one run from setup to Scorecard. The module owns
input validation, driving the interview graph, the Transcript, and saving
after every Turn. Callers (the WebSocket route, tests) see only `start`,
`answer` and `end`, each an async iterator of typed events.
"""

import asyncio
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
    session_id: str


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
class Resumed:
    """Sent first when a client re-attaches: everything said so far, client-ready."""
    session_id: str
    messages: list[dict]  # {"type": "agent"|"user", "source", "content", "metadata"}
    num_questions: int
    questions_answered: int


@dataclass(frozen=True)
class Error:
    content: str
    fatal: bool = False  # fatal: the session cannot continue
    code: str = ""


Event = Started | Typing | AgentMessage | WaitingForInput | Summary | Complete | Resumed | Error


class InterviewSession:
    def __init__(self, graph, store=database):
        self._graph = graph
        self._store = store
        self.id = str(uuid.uuid4())
        # new | in_progress | completed | failed | abandoned
        self.status = "new"
        self.questions_asked = 0
        self.transcript: list[dict] = []
        self._saved = 0  # transcript messages already persisted
        self._replay: list[dict] = []  # client-ready history for re-attaching
        self._awaiting = False
        self._summary: Summary | None = None
        self._num_questions = settings.num_questions
        self._sub: asyncio.Queue | None = None  # events for the attached client
        self._task: asyncio.Task | None = None  # the turn in flight
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
            self.status = "failed"
            yield Error("Could not start the interview. Please try again.", fatal=True)
            return
        self.status = "in_progress"

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
        queue = self._begin_turn(initial_state)
        yield Started(
            content=f"Starting {interview_type} interview for {job_position}...",
            num_questions=self._num_questions,
            session_id=self.id,
        )
        async for event in self._stream(queue):
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
        async for event in self._stream(self._begin_turn(Command(resume=text))):
            yield event

    async def end(self) -> AsyncIterator[Event]:
        if not self._awaiting:
            yield Error("The interview is not waiting for an answer.")
            return
        logger.info("[%s] User requested end of interview", self.id)
        async for event in self._stream(self._begin_turn(Command(resume=END_SENTINEL))):
            yield event

    async def resume(self) -> AsyncIterator[Event]:
        """Re-attach a new client: replay what was said, then rejoin the live turn.

        Replaces any previously attached client, whose event stream ends.
        """
        queue = self._attach()
        # Snapshot in the same synchronous step as attaching, so each event
        # lands in exactly one of: the snapshot, or the queue.
        history = list(self._replay)
        running = self._task is not None and not self._task.done()
        yield Resumed(self.id, history, self._num_questions, self.questions_asked)

        if self.status == "completed" and self._summary:
            yield self._summary
            yield Complete()
        elif running:
            yield Typing()
            async for event in self._stream(queue):
                yield event
        elif self.status == "in_progress" and self._awaiting:
            yield WaitingForInput()
        else:
            yield Error("This interview can no longer continue.", fatal=True)

    def detach(self) -> None:
        """The client went away. The turn in flight keeps running and saving."""
        self._sub = None

    async def idle(self) -> None:
        """Wait for the turn in flight, if any, to finish."""
        if self._task is not None:
            await asyncio.wait([self._task])

    async def abandon(self) -> None:
        """Give up on this session: cancel any turn and mark it abandoned."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.wait([self._task])
        if self.status == "in_progress":
            self.status = "abandoned"
            self._awaiting = False
            await self._mark_abandoned()
            self._forget_thread()

    # --- Internals ---

    def _attach(self) -> asyncio.Queue:
        if self._sub is not None:
            self._sub.put_nowait(None)  # end the previous client's stream
        self._sub = asyncio.Queue()
        return self._sub

    def _emit(self, event: Event | None) -> None:
        if self._sub is not None:
            self._sub.put_nowait(event)

    @staticmethod
    async def _stream(queue: asyncio.Queue) -> AsyncIterator[Event]:
        while (event := await queue.get()) is not None:
            yield event

    def _begin_turn(self, graph_input) -> asyncio.Queue:
        queue = self._attach()
        self._awaiting = False
        self._task = asyncio.create_task(self._produce(graph_input))
        return queue

    async def _produce(self, graph_input) -> None:
        """Run the graph until it pauses for input or finishes; save the result.

        Runs as its own task so a dropped client does not cancel an LLM call
        that is already running.
        """
        self._emit(Typing())
        hit_interrupt = False
        try:
            stream = self._graph.astream(graph_input, config=self._config, stream_mode="updates")
            async for chunk in stream:
                for node_name, updates in chunk.items():
                    if node_name == "__interrupt__":
                        hit_interrupt = True
                    elif isinstance(updates, dict):
                        for event in self._events_from_update(updates):
                            self._emit(event)
            await self._save_progress()
            if not hit_interrupt:
                await self._store.finish_interview(
                    self.id, self.questions_asked, self._summary.data if self._summary else None
                )
        except Exception:
            logger.error("[%s] Turn failed", self.id, exc_info=True)
            self.status = "failed"
            await self._mark_abandoned()
            self._forget_thread()
            self._emit(Error("Something went wrong. Please start a new interview.", fatal=True))
        else:
            if hit_interrupt:
                self._awaiting = True
                self._emit(WaitingForInput())
            else:
                self.status = "completed"
                logger.info("[%s] Interview complete", self.id)
                self._forget_thread()
                self._emit(Complete())
        finally:
            self._emit(None)

    def _events_from_update(self, updates: dict) -> list[Event]:
        events: list[Event] = []
        for msg in updates.get("messages", []):
            if isinstance(msg, AIMessage) and msg.name:
                self.transcript.append({"role": "assistant", "name": msg.name, "content": msg.content})
                if msg.name == "summary":
                    self._summary = Summary(msg.content, updates.get("summary_data") or {})
                    events.append(self._summary)
                else:
                    event = self._agent_message(msg, updates)
                    events.append(event)
                    metadata = {}
                    if event.score is not None:
                        metadata = {"score": event.score, "tier": event.tier, "tier_index": event.tier_index}
                    self._replay.append({
                        "type": "agent", "source": event.source,
                        "content": event.content, "metadata": metadata,
                    })
            elif isinstance(msg, HumanMessage):
                self.transcript.append({"role": "user", "name": getattr(msg, "name", ""), "content": msg.content})
                if not updates.get("end_requested"):
                    self._replay.append({"type": "user", "content": msg.content})
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

    async def _mark_abandoned(self) -> None:
        try:
            await self._store.mark_abandoned(self.id)
        except Exception:
            logger.error("[%s] Failed to mark interview abandoned", self.id, exc_info=True)

    def _forget_thread(self) -> None:
        """Drop the graph checkpoint; this session can no longer resume."""
        checkpointer = getattr(self._graph, "checkpointer", None)
        if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
            try:
                checkpointer.delete_thread(self.id)
            except Exception:
                logger.warning("[%s] Could not delete graph checkpoint", self.id, exc_info=True)
