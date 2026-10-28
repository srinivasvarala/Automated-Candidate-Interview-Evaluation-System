"""Prep plan module tests: scripted LLM, no network."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app import prep
from app.llm_service import LLMOutputError

pytestmark = pytest.mark.asyncio

PLAN = prep.PlanInput(role="AI Engineer", seniority="senior")


def outline_json(extra=(), drop=(), titles=True):
    areas = [
        {"key": key, "title": title if titles else "", "case": f"Case for {key}", "goal": f"Goal for {key}"}
        for key, title in prep.BACKBONE
        if key not in drop
    ]
    areas += [{"key": key, "title": key.title(), "case": "c", "goal": "g"} for key in extra]
    return json.dumps({"areas": areas})


LESSON = {
    "case_title": "RAG support bot",
    "scenario": "10,000 documents.",
    "design": "1. Ingest\n2. Retrieve\n3. Generate",
    "trade_offs": "Hybrid vs dense.",
    "failure_modes": "Stale index.",
    "testing_and_observability": "Eval set; trace latency.",
    "interview_angle": "Lead with requirements.",
    "checks": ["How would you evaluate it?", "What breaks at 10x?"],
}
AREA = prep.OutlineArea(key="architecture", title="Architecture", case="RAG support bot", goal="Design it")


# --- input validation ---

async def test_validate_defaults_seniority_to_mid():
    assert prep.validate_plan_input("  AI Engineer ").seniority == "mid"


async def test_validate_normalises_and_truncates():
    plan = prep.validate_plan_input("SRE", "Senior", " jd ", "x" * 20_000)
    assert (plan.role, plan.seniority, plan.job_description) == ("SRE", "senior", "jd")
    assert len(plan.resume_text) == prep.MAX_RESUME


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"role": "  "}, "required"),
        ({"role": "x" * 201}, "200"),
        ({"role": "SRE", "seniority": "wizard"}, "Seniority"),
        ({"role": "SRE", "job_description": "x" * 10_001}, "10,000"),
    ],
)
async def test_validate_rejects_bad_input(kwargs, message):
    with pytest.raises(prep.PrepInputError, match=message):
        prep.validate_plan_input(**kwargs)


# --- outline ---

async def test_outline_has_every_backbone_area_in_order():
    with patch("app.llm_service.complete_json", AsyncMock(return_value=outline_json())):
        outline = await prep.generate_outline(PLAN)
    assert [a.key for a in outline.areas] == [key for key, _ in prep.BACKBONE]


async def test_outline_uses_canonical_titles_and_keeps_extras_last():
    reply = outline_json(extra=["evaluation"], titles=False)
    with patch("app.llm_service.complete_json", AsyncMock(return_value=reply)):
        outline = await prep.generate_outline(PLAN)
    assert outline.areas[0].title == "Architecture"
    assert [a.key for a in outline.areas][-1] == "evaluation"
    assert outline.areas[-1].title == "Evaluation"


async def test_outline_missing_area_is_retried_then_fails():
    mock = AsyncMock(return_value=outline_json(drop=["observability"]))
    with patch("app.llm_service.complete_json", mock), pytest.raises(LLMOutputError):
        await prep.generate_outline(PLAN)
    assert mock.call_count == 2


async def test_outline_too_many_extras_is_rejected():
    mock = AsyncMock(return_value=outline_json(extra=["a", "b", "c"]))
    with patch("app.llm_service.complete_json", mock), pytest.raises(LLMOutputError):
        await prep.generate_outline(PLAN)


async def test_outline_recovers_on_retry():
    mock = AsyncMock(side_effect=["not json", outline_json()])
    with patch("app.llm_service.complete_json", mock):
        outline = await prep.generate_outline(PLAN)
    assert len(outline.areas) == len(prep.BACKBONE)


async def test_outline_prompt_carries_role_seniority_and_job_description():
    plan = prep.PlanInput("AI Architect", "staff", "Build agent platforms on Kubernetes", "")
    mock = AsyncMock(return_value=outline_json())
    with patch("app.llm_service.complete_json", mock):
        await prep.generate_outline(plan)
    system, user = mock.call_args[0][0]
    assert "Observability" in system["content"]
    assert "AI Architect" in user["content"] and "staff" in user["content"]
    assert "agent platforms on Kubernetes" in user["content"]
    assert "reference architectures" in user["content"]  # the role template hint


# --- lesson ---

async def test_lesson_is_typed_and_gets_the_long_timeout():
    mock = AsyncMock(return_value=json.dumps(LESSON))
    with patch("app.llm_service.complete_json", mock):
        lesson = await prep.generate_lesson(PLAN, AREA)
    assert lesson.case_title == "RAG support bot"
    assert lesson.checks == LESSON["checks"]
    assert mock.call_args.kwargs["timeout"] == prep.LESSON_TIMEOUT
    assert "RAG support bot" in mock.call_args[0][0][0]["content"]


@pytest.mark.parametrize(
    "broken",
    [
        {**LESSON, "checks": ["only one"]},
        {**LESSON, "checks": ["a", "b", "c"]},
        {**LESSON, "design": "  "},
        {k: v for k, v in LESSON.items() if k != "failure_modes"},
    ],
)
async def test_invalid_lesson_is_retried_then_fails(broken):
    mock = AsyncMock(return_value=json.dumps(broken))
    with patch("app.llm_service.complete_json", mock), pytest.raises(LLMOutputError):
        await prep.generate_lesson(PLAN, AREA)
    assert mock.call_count == 2


# --- checks ---

LESSON_OBJ = prep.LessonContent(**LESSON)


async def test_check_gets_feedback_and_a_score():
    with patch("app.llm_service.complete", AsyncMock(return_value="Good. Add failure modes.")), \
         patch("app.llm_service.complete_json", AsyncMock(return_value='{"score": 8}')):
        result = await prep.score_check(PLAN, AREA, LESSON_OBJ, "How would you evaluate it?", "Golden set plus online metrics.")
    assert result == prep.CheckResult("Good. Add failure modes.", 8)


async def test_unscorable_check_keeps_the_feedback():
    json_mock = AsyncMock(return_value="garbage")
    with patch("app.llm_service.complete", AsyncMock(return_value="Solid answer.")), \
         patch("app.llm_service.complete_json", json_mock):
        result = await prep.score_check(PLAN, AREA, LESSON_OBJ, "q", "a")
    assert result == prep.CheckResult("Solid answer.", None)
    assert json_mock.call_count == 2  # one retry


async def test_empty_feedback_gets_a_fallback():
    with patch("app.llm_service.complete", AsyncMock(return_value="  ")), \
         patch("app.llm_service.complete_json", AsyncMock(return_value='{"score": 5}')):
        result = await prep.score_check(PLAN, AREA, LESSON_OBJ, "q", "a")
    assert result.feedback and result.score == 5


async def test_role_templates_list_quick_picks():
    templates = prep.role_templates()
    assert {"role": "AI Engineer", "focus": prep.ROLE_TEMPLATES["AI Engineer"]} in templates


# --- mock focus ---

def _area(key, title, scores=(), case=None):
    return {
        "key": key, "title": title, "case": case or f"case for {key}",
        "checks": [{"idx": i, "score": s} for i, s in enumerate(scores)],
    }


def _plan(*areas):
    return {"role": "AI Engineer", "seniority": "senior", "areas": list(areas)}


async def test_focus_without_checks_lists_every_area_in_plan_order():
    focus = prep.mock_focus(_plan(_area("a", "Architecture"), _area("b", "Design")))
    assert focus.splitlines() == [
        "Role: AI Engineer (senior).",
        "Areas in the study plan:",
        "- Architecture: case for a",
        "- Design: case for b",
    ]


async def test_focus_puts_the_weakest_scored_areas_first():
    plan = _plan(
        _area("a", "Architecture", [9, 9]),
        _area("b", "Design", [3, 5]),
        _area("c", "Testing"),  # never answered: not weak, just not practised
        _area("d", "Observability", [2]),
    )
    lines = prep.mock_focus(plan).splitlines()
    assert lines[1] == "Weakest areas so far (ask about these first):"
    assert lines[2].startswith("- Observability:") and "2.0/10" in lines[2]
    assert lines[3].startswith("- Design:") and "4.0/10" in lines[3]
    assert lines[4].startswith("- Architecture:") and "9.0/10" in lines[4]
    assert lines[5] == "Other areas in the study plan:"
    assert lines[6:] == ["- Testing: case for c"]


async def test_focus_names_at_most_three_weak_areas():
    plan = _plan(*[_area(k, k.upper(), [s]) for k, s in zip("abcde", (8, 1, 7, 2, 3))])
    lines = prep.mock_focus(plan).splitlines()
    weak = lines[2:lines.index("Other areas in the study plan:")]
    assert [w.split(":")[0] for w in weak] == ["- B", "- D", "- E"]
    assert [line.split(":")[0] for line in lines[lines.index("Other areas in the study plan:") + 1:]] == ["- A", "- C"]


async def test_focus_ignores_unscored_checks_and_breaks_ties_by_plan_order():
    plan = _plan(
        _area("a", "A", [5]),
        _area("b", "B", [5]),
        {"key": "c", "title": "C", "case": "x", "checks": [{"idx": 0, "score": None}]},
    )
    lines = prep.mock_focus(plan).splitlines()
    assert [line.split(":")[0] for line in lines[2:4]] == ["- A", "- B"]
    assert lines[-1] == "- C: x"
    assert "practice check average" not in lines[-1]
