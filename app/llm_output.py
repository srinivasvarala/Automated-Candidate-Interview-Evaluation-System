"""Typed shapes of the JSON the model is asked to return."""

from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError, field_validator

_SCORE = TypeAdapter(int)


def _valid_score(value: Any) -> int | None:
    try:
        score = _SCORE.validate_python(value)
    except ValidationError:
        return None
    return score if 1 <= score <= 10 else None


class AnswerScore(BaseModel):
    """The Evaluator's Score for one answer."""
    score: int

    @field_validator("score")
    @classmethod
    def _in_range(cls, value: int) -> int:
        if not 1 <= value <= 10:
            raise ValueError("score must be between 1 and 10")
        return value


class ResumeAnalysis(BaseModel):
    verified_skills: list[str] = []
    gaps: list[str] = []
    surprises: list[str] = []


class ScorecardOutput(BaseModel):
    """The model's end-of-session Scorecard, before the difficulty curve is added."""
    scores: dict[str, int]
    strengths: list[str] = []
    improvements: list[str] = []
    overall_summary: str = ""
    resume_analysis: ResumeAnalysis | None = None

    @field_validator("scores", mode="before")
    @classmethod
    def _keep_valid_categories(cls, value: Any) -> dict[str, int]:
        if not isinstance(value, dict):
            raise ValueError("scores must be an object")
        kept = {name: s for name, raw in value.items() if (s := _valid_score(raw)) is not None}
        if not kept:
            raise ValueError("scores has no valid category score")
        return kept
