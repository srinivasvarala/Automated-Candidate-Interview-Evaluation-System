import pytest

from app import database

pytestmark = pytest.mark.asyncio

OUTLINE = [
    {"key": "architecture", "title": "Architecture", "case": "RAG bot", "goal": "Design it"},
    {"key": "testing", "title": "Testing", "case": "Eval harness", "goal": "Test it"},
]
LESSON = {"case_title": "RAG bot", "checks": ["q1", "q2"]}


async def make_plan(plan_id="p1", role="AI Engineer"):
    await database.create_prep_plan(plan_id, role, "senior", "the jd", "the cv", OUTLINE)


async def test_new_plan_has_every_area_not_started():
    await make_plan()
    plan = await database.get_prep_plan("p1")
    assert (plan["role"], plan["seniority"], plan["job_description"], plan["resume_text"]) == (
        "AI Engineer", "senior", "the jd", "the cv")
    assert [(a["key"], a["status"], a["lesson"], a["checks"]) for a in plan["areas"]] == [
        ("architecture", "not_started", None, []),
        ("testing", "not_started", None, []),
    ]
    assert plan["areas"][0]["case"] == "RAG bot"


async def test_unknown_plan_is_none():
    assert await database.get_prep_plan("nope") is None


async def test_lesson_is_saved_and_listed_as_ready():
    await make_plan()
    assert await database.save_lesson("p1", "architecture", LESSON) == LESSON
    area = (await database.get_prep_plan("p1"))["areas"][0]
    assert area["status"] == "ready"
    assert area["lesson"] == LESSON


async def test_first_saved_lesson_wins():
    await make_plan()
    await database.save_lesson("p1", "architecture", LESSON)
    stored = await database.save_lesson("p1", "architecture", {"case_title": "other", "checks": []})
    assert stored == LESSON


async def test_checks_are_stored_in_order_and_replaced_on_reanswer():
    await make_plan()
    await database.save_lesson("p1", "architecture", LESSON)
    await database.save_check("p1", "architecture", 1, "q2", "a2", None, "fb2")
    await database.save_check("p1", "architecture", 0, "q1", "first", 4, "fb1")
    await database.save_check("p1", "architecture", 0, "q1", "second", 8, "better")
    checks = (await database.get_prep_plan("p1"))["areas"][0]["checks"]
    assert [(c["idx"], c["answer"], c["score"], c["feedback"]) for c in checks] == [
        (0, "second", 8, "better"),
        (1, "a2", None, "fb2"),  # an unscored check keeps its feedback
    ]


async def test_complete_lesson():
    await make_plan()
    assert await database.complete_lesson("p1", "architecture") is False  # nothing generated yet
    await database.save_lesson("p1", "architecture", LESSON)
    assert await database.complete_lesson("p1", "architecture") is True
    assert await database.complete_lesson("p1", "architecture") is True  # idempotent
    assert (await database.get_prep_plan("p1"))["areas"][0]["status"] == "done"


async def test_list_shows_progress_counts_newest_first():
    await make_plan("old", "SRE")
    await make_plan("new", "AI Architect")
    await database.save_lesson("new", "architecture", LESSON)
    await database.save_lesson("new", "testing", LESSON)
    await database.complete_lesson("new", "testing")

    plans = await database.list_prep_plans()
    assert [p["id"] for p in plans] == ["new", "old"]
    assert (plans[0]["areas_total"], plans[0]["lessons_generated"], plans[0]["lessons_done"]) == (2, 2, 1)
    assert (plans[1]["lessons_generated"], plans[1]["lessons_done"]) == (0, 0)
    assert "resume_text" not in plans[0] and "outline" not in plans[0]


async def test_list_paginates():
    for i in range(3):
        await make_plan(f"p{i}")
    assert [p["id"] for p in await database.list_prep_plans(limit=2)] == ["p2", "p1"]
    assert [p["id"] for p in await database.list_prep_plans(limit=2, offset=2)] == ["p0"]


async def test_activity_moves_a_plan_updated_at_forward():
    await make_plan()
    before = (await database.get_prep_plan("p1"))["updated_at"]
    await database.save_lesson("p1", "architecture", LESSON)
    assert (await database.get_prep_plan("p1"))["updated_at"] > before


async def test_plans_do_not_share_lessons_or_checks():
    await make_plan("a")
    await make_plan("b")
    await database.save_lesson("a", "architecture", LESSON)
    await database.save_check("a", "architecture", 0, "q", "ans", 5, "fb")
    other = (await database.get_prep_plan("b"))["areas"][0]
    assert (other["status"], other["checks"]) == ("not_started", [])


async def test_prep_tables_survive_init_on_an_existing_database():
    await make_plan()
    await database.init_db()  # idempotent
    assert await database.get_prep_plan("p1") is not None
