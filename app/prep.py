"""Prep plan module.

Preparation before the mock interview: an outline of Areas for a role and
Seniority, a Lesson per Area generated on demand, and Checks that get feedback
and a Score. Callers (the REST routes, tests) see only `validate_plan_input`,
`generate_outline`, `generate_lesson` and `score_check`; every model reply is
validated by `llm_service.complete_model`, so a bad reply is retried once and
then raises `LLMOutputError` rather than being papered over.
"""

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

from app import llm_service
from app.llm_output import AnswerScore
from app.llm_service import LLMOutputError

logger = logging.getLogger(__name__)

MAX_ROLE = 200
MAX_JOB_DESCRIPTION = 10_000
MAX_RESUME = 10_000
MAX_EXTRA_AREAS = 2
LESSON_TIMEOUT = 120  # seconds; a Lesson is a long generation

SENIORITIES = ("junior", "mid", "senior", "staff")
DEFAULT_SENIORITY = "mid"

# (key, title): every Prep plan covers all of these, in this order.
BACKBONE = (
    ("architecture", "Architecture"),
    ("design", "Design"),
    ("frameworks", "Frameworks and tooling"),
    ("development", "Development"),
    ("testing", "Testing"),
    ("deployment", "Deployment"),
    ("observability", "Observability"),
    ("security", "Security"),
    ("data", "Data"),
    ("cost_performance", "Cost and performance"),
)
_BACKBONE_TITLES = dict(BACKBONE)

_SENIORITY_GUIDANCE = {
    "junior": "Explain the reasoning behind each choice; assume solid fundamentals but little production experience. Favour one clear, well-trodden design over breadth.",
    "mid": "Assume the candidate has shipped features and wants to own a service end to end. Cover the main trade-offs and the common failure modes.",
    "senior": "Assume the candidate designs systems others build on. Go deep on trade-offs, failure modes, scale limits, and how to justify decisions to stakeholders.",
    "staff": "Assume the candidate sets technical direction across teams. Emphasise ambiguity, cross-cutting concerns, build-versus-buy, migration paths, and organisational trade-offs.",
}

# Quick-pick roles for the UI; the hint steers emphasis for that role.
ROLE_TEMPLATES = {
    "AI Engineer": "Building LLM applications: retrieval-augmented generation, agents and tool use, prompt and model versioning, evaluation, guardrails, serving and latency.",
    "AI Architect": "Designing AI platforms and reference architectures: model and vendor selection, data and evaluation pipelines, governance, cost, multi-team reuse.",
    "ML Engineer": "Training and serving models in production: feature pipelines, experiment tracking, deployment, monitoring drift, retraining.",
    "Backend Engineer": "Designing and operating services: APIs, data stores, queues, caching, reliability, scaling.",
    "Data Engineer": "Building data platforms: ingestion, batch and streaming pipelines, modelling, quality, orchestration, cost.",
    "Platform / DevOps Engineer": "Building the platform others deploy on: CI/CD, infrastructure as code, Kubernetes, observability, incident response, cost control.",
}


class PrepInputError(ValueError):
    """The plan inputs are not acceptable; the message is safe to show the user."""


@dataclass(frozen=True)
class PlanInput:
    role: str
    seniority: str = DEFAULT_SENIORITY
    job_description: str = ""
    resume_text: str = ""


def validate_plan_input(
    role: str, seniority: str = "", job_description: str = "", resume_text: str = ""
) -> PlanInput:
    role = (role or "").strip()
    if not role:
        raise PrepInputError("Role is required.")
    if len(role) > MAX_ROLE:
        raise PrepInputError(f"Role must be under {MAX_ROLE} characters.")
    seniority = (seniority or "").strip().lower() or DEFAULT_SENIORITY
    if seniority not in SENIORITIES:
        raise PrepInputError(f"Seniority must be one of: {', '.join(SENIORITIES)}.")
    job_description = (job_description or "").strip()
    if len(job_description) > MAX_JOB_DESCRIPTION:
        raise PrepInputError("Job description must be under 10,000 characters.")
    return PlanInput(role, seniority, job_description, (resume_text or "").strip()[:MAX_RESUME])


# --- Typed shapes of the model's replies ---

class OutlineArea(BaseModel):
    key: str
    title: str = ""
    case: str  # the concrete real-world case this Area's Lesson will walk through
    goal: str  # one line: what the candidate should be able to do afterwards

    @field_validator("key", "case", "goal")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class Outline(BaseModel):
    areas: list[OutlineArea]

    @model_validator(mode="after")
    def _backbone_complete(self) -> "Outline":
        by_key: dict[str, OutlineArea] = {}
        for area in self.areas:
            if area.key in by_key:
                raise ValueError(f"duplicate area {area.key!r}")
            by_key[area.key] = area
        missing = [key for key, _ in BACKBONE if key not in by_key]
        if missing:
            raise ValueError(f"missing backbone areas: {', '.join(missing)}")
        extras = [a for a in self.areas if a.key not in _BACKBONE_TITLES]
        if len(extras) > MAX_EXTRA_AREAS:
            raise ValueError(f"at most {MAX_EXTRA_AREAS} role-specific areas")
        for extra in extras:
            extra.title = extra.title.strip() or extra.key.replace("_", " ").capitalize()
        # canonical order and titles for the backbone, extras last
        ordered = []
        for key, title in BACKBONE:
            area = by_key[key]
            area.title = title
            ordered.append(area)
        self.areas = ordered + extras
        return self


class LessonContent(BaseModel):
    """One Lesson. Every field except `checks` is markdown."""
    case_title: str
    scenario: str
    design: str
    trade_offs: str
    failure_modes: str
    testing_and_observability: str
    interview_angle: str
    checks: list[str]

    @field_validator("case_title", "scenario", "design", "trade_offs", "failure_modes",
                     "testing_and_observability", "interview_angle")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("checks")
    @classmethod
    def _two_checks(cls, value: list[str]) -> list[str]:
        value = [q.strip() for q in value if isinstance(q, str) and q.strip()]
        if len(value) != 2:
            raise ValueError("exactly two checks are required")
        return value


@dataclass(frozen=True)
class CheckResult:
    feedback: str
    score: int | None  # None when the model's score could not be parsed


# --- Prompts ---

def _context(plan: PlanInput) -> str:
    parts = [f"Role: {plan.role}", f"Seniority: {plan.seniority}. {_SENIORITY_GUIDANCE[plan.seniority]}"]
    hint = next((h for name, h in ROLE_TEMPLATES.items() if name.lower() == plan.role.lower()), "")
    if hint:
        parts.append(f"Typical focus for this role: {hint}")
    if plan.job_description:
        parts.append(f"Job description (tailor to its skills, tools and responsibilities):\n{plan.job_description[:3000]}")
    if plan.resume_text:
        parts.append(
            "Candidate's resume (skip basics they clearly already have; probe likely gaps):\n"
            f"{plan.resume_text[:4000]}"
        )
    return "\n\n".join(parts)


def _outline_messages(plan: PlanInput) -> list[dict]:
    areas = "\n".join(f"- {key}: {title}" for key, title in BACKBONE)
    system = f"""You are a staff-level engineer building an interview-preparation plan. The plan teaches how to design, build, ship and operate real-world applications, not theory.

Produce one Area for each of these topics, in this order, using exactly these keys:
{areas}

For every Area choose ONE concrete real-world case that fits the role (for example "A RAG support bot answering questions over 10,000 internal documents") and write a one-line goal: what the candidate should be able to do afterwards. Cases should differ enough that each Area teaches something new.

You may add up to {MAX_EXTRA_AREAS} extra Areas that are specific to this role (for example evaluation for an AI engineer). Give each its own snake_case key and a title. Never drop or rename the topics above.

Return ONLY valid JSON:
{{"areas": [{{"key": "architecture", "title": "Architecture", "case": "...", "goal": "..."}}]}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": _context(plan)}]


def _lesson_messages(plan: PlanInput, area: OutlineArea) -> list[dict]:
    system = f"""You are a staff-level engineer writing one lesson of an interview-preparation plan. The lesson teaches how to design and build a real application, not theory, and how to talk about it in an interview.

Area: {area.title}
Case: {area.case}
Goal: {area.goal}

Walk through the case as a real project. Name concrete technologies, patterns and realistic numbers, and say when NOT to use each choice. Never invent library APIs or version-specific details; if unsure, describe the concept and say what to check in the documentation. No filler. Keep the whole lesson to roughly 1,000 to 1,300 words across all fields: choose the most important points rather than covering everything.

Write each field as markdown:
- case_title: a short title for the case.
- scenario: the problem, who uses it, and the requirements and constraints (scale, latency, budget).
- design: the components and the data flow as numbered steps, and the key decisions.
- trade_offs: the main alternatives considered and why one was chosen.
- failure_modes: what breaks in production and how to detect and handle it.
- testing_and_observability: how to test it before release and what to measure and alert on afterwards (deployment and cost considerations belong here when they matter).
- interview_angle: how to explain this in an interview, what a strong answer includes, and common mistakes.
- checks: exactly two short questions that test whether the reader could explain or adapt this design.

Return ONLY valid JSON with exactly those keys; "checks" is a list of two strings."""
    return [{"role": "system", "content": system}, {"role": "user", "content": _context(plan)}]


# --- Interface ---

async def generate_outline(plan: PlanInput) -> Outline:
    """One call: the Areas of a Prep plan. Raises LLMOutputError if the reply can't be validated."""
    logger.info("Generating outline: role=%s, seniority=%s", plan.role, plan.seniority)
    return await llm_service.complete_model(_outline_messages(plan), Outline, temperature=0.5)


async def generate_lesson(plan: PlanInput, area: OutlineArea) -> LessonContent:
    """One call: the Lesson for one Area. Raises LLMOutputError if the reply can't be validated."""
    logger.info("Generating lesson: role=%s, area=%s", plan.role, area.key)
    return await llm_service.complete_model(
        _lesson_messages(plan, area), LessonContent, temperature=0.5, timeout=LESSON_TIMEOUT
    )


async def score_check(plan: PlanInput, area: OutlineArea, lesson: LessonContent, question: str, answer: str) -> CheckResult:
    """Feedback on one Check answer, plus a Score. An unparseable Score leaves feedback with no Score."""
    feedback_prompt = f"""You are a career coach giving feedback on an answer to a practice question from a {plan.role} interview-preparation lesson ({plan.seniority} seniority).

Lesson case: {lesson.case_title}
Key points the lesson teaches:
{lesson.design[:1500]}

Give brief, constructive feedback on the answer (2-4 sentences): one strength and one thing to improve, tied to real-world design and operations. Do not ask any questions and do not repeat the question."""
    messages = [
        {"role": "system", "content": feedback_prompt},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
    ]
    feedback = (await llm_service.complete(messages)).strip()
    if not feedback:
        feedback = "Thanks for answering. Try to be more specific about trade-offs and failure modes."

    score_prompt = f"""Rate the answer on a scale of 1-10 for a {plan.seniority} {plan.role}.

Consider: accuracy, depth, specificity, trade-offs and operational awareness.
- 1-3: Weak (vague, incorrect, or off-topic)
- 4-6: Adequate (partially correct, lacks depth or examples)
- 7-8: Strong (specific, well-structured, shows real understanding)
- 9-10: Exceptional (insightful, goes beyond the question)

Return ONLY a JSON object: {{"score": <1-10>}}"""
    score_messages = [
        {"role": "system", "content": score_prompt},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
    ]
    score: int | None
    try:
        score = (await llm_service.complete_model(score_messages, AnswerScore)).score
    except LLMOutputError:
        logger.warning("Check score unavailable; returning feedback only")
        score = None
    return CheckResult(feedback, score)


MAX_WEAK_AREAS = 3


def mock_focus(plan: dict) -> str:
    """The prep focus for a mock interview, from a stored plan (as `database.get_prep_plan` returns it).

    The weakest Areas by average Check Score come first (at most MAX_WEAK_AREAS, only Areas with
    a Score), then every other Area in plan order.
    """
    scored = []
    for position, area in enumerate(plan["areas"]):
        scores = [c["score"] for c in area.get("checks", []) if isinstance(c.get("score"), int)]
        if scores:
            scored.append((sum(scores) / len(scores), position, area))
    weak = [item for item in sorted(scored)[:MAX_WEAK_AREAS]]
    weak_keys = {area["key"] for _, _, area in weak}

    lines = [f"Role: {plan['role']} ({plan['seniority']})."]
    if weak:
        lines.append("Weakest areas so far (ask about these first):")
        lines += [f"- {a['title']}: {a['case']} (practice check average {avg:.1f}/10)" for avg, _, a in weak]
        lines.append("Other areas in the study plan:")
    else:
        lines.append("Areas in the study plan:")
    lines += [f"- {a['title']}: {a['case']}" for a in plan["areas"] if a["key"] not in weak_keys]
    return "\n".join(lines)


def role_templates() -> list[dict[str, Any]]:
    """Quick-pick roles for the setup screen."""
    return [{"role": name, "focus": hint} for name, hint in ROLE_TEMPLATES.items()]
