from __future__ import annotations

from typing import Annotated, Any
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# --- LangGraph State ---

class InterviewState:
    """TypedDict-style state for LangGraph (defined as annotations)."""
    pass


# We use a plain TypedDict for LangGraph compatibility
from typing import TypedDict


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    job_position: str
    questions_asked: int
    current_phase: str  # "interviewer" | "candidate" | "evaluator" | "summary"
    end_requested: bool
    summary_data: dict[str, Any] | None


# --- WebSocket Message Models ---

class WSMessage(BaseModel):
    type: str
    content: str = ""
    source: str = ""
    event: str = ""
    metadata: dict[str, Any] = {}


class Scorecard(BaseModel):
    technical: int = 0
    problem_solving: int = 0
    communication: int = 0
    culture_fit: int = 0


class SummaryData(BaseModel):
    scores: Scorecard = Scorecard()
    strengths: list[str] = []
    improvements: list[str] = []
    overall_summary: str = ""
