from __future__ import annotations

from typing import Annotated, Any
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict


# --- LangGraph State ---

class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    job_position: str
    job_description: str
    resume_text: str  # extracted text from uploaded resume (optional)
    interview_type: str  # "mixed" | "behavioral" | "technical" | "system_design"
    questions_asked: int
    current_phase: str  # "interviewer" | "candidate" | "evaluator" | "summary"
    end_requested: bool
    summary_data: dict[str, Any] | None
    prep_focus: str  # study-plan context for the Interviewer (optional)
    scores: list[int | None]  # Evaluator score (1-10) per answered question; None if unscored


# --- WebSocket Message Models ---

class WSMessage(BaseModel):
    type: str
    content: str = ""
    source: str = ""
    event: str = ""
    metadata: dict[str, Any] = {}
