"""Interview session tests: real graph, scripted LLM, temp SQLite."""

import asyncio
from unittest.mock import patch

import pytest

from app import database, session
from app.graph import build_interview_graph

pytestmark = pytest.mark.asyncio

SUMMARY_JSON = (
    '{"scores": {"technical": 7, "problem_solving": 8, "communication": 6, "culture_fit": 7},'
    ' "strengths": ["a"], "improvements": ["b"], "overall_summary": "ok"}'
)


async def fake_complete(messages, temperature=0.7):
    if "career coach" in messages[0]["content"]:
        return "Solid answer. 💡 **Stronger answer:** Be more specific."
    return "Describe a hard bug you fixed."


async def fake_complete_json(messages, temperature=0.3):
    if "Rate the candidate" in messages[0]["content"]:
        return '{"score": 8}'
    return SUMMARY_JSON


@pytest.fixture(autouse=True)
def fake_llm():
    with patch("app.agents.llm_service.complete", fake_complete), \
         patch("app.agents.llm_service.complete_json", fake_complete_json):
        yield


@pytest.fixture
def interview():
    return session.InterviewSession(build_interview_graph())


async def collect(events):
    return [e async for e in events]


def kinds(events):
    return [type(e).__name__ for e in events]


async def test_start_streams_first_question_and_waits(interview):
    events = await collect(interview.start("Software Engineer", interview_type="technical"))
    assert kinds(events) == ["Started", "Typing", "AgentMessage", "WaitingForInput"]
    assert events[2].source == "interviewer"
    assert interview.awaiting_input


async def test_full_interview_to_scorecard(interview):
    await collect(interview.start("Software Engineer"))
    turn = await collect(interview.answer("I fixed a race condition in our queue."))
    assert kinds(turn) == ["Typing", "AgentMessage", "AgentMessage", "WaitingForInput"]
    evaluator = turn[1]
    assert (evaluator.source, evaluator.score, evaluator.tier) == ("evaluator", 8, "Warm-up")

    final = await collect(interview.end())
    assert kinds(final) == ["Typing", "Summary", "Complete"]
    assert final[1].data["scores"]["technical"] == 7
    assert "difficulty_curve" in final[1].data
    assert not interview.awaiting_input

    saved = await database.get_interview(interview.id)
    assert saved["status"] == "completed"
    assert saved["questions_asked"] == 1
    assert saved["summary_data"]["scores"]["technical"] == 7
    assert [m["name"] for m in saved["messages"]] == [
        "interviewer", "candidate", "evaluator", "interviewer", "candidate", "summary",
    ]


async def test_progress_is_saved_after_every_turn(interview):
    await collect(interview.start("Software Engineer"))
    saved = await database.get_interview(interview.id)
    assert saved["status"] == "in_progress"
    assert len(saved["messages"]) == 1

    await collect(interview.answer("My answer"))
    saved = await database.get_interview(interview.id)
    assert saved["status"] == "in_progress"
    assert saved["questions_asked"] == 1
    assert len(saved["messages"]) == 4  # question, answer, feedback, next question


@pytest.mark.parametrize("kwargs,fragment", [
    ({"job_position": "  "}, "required"),
    ({"job_position": "x" * 201}, "200"),
    ({"job_position": "SWE", "job_description": "x" * 10_001}, "10,000"),
])
async def test_start_validation_errors_are_fatal_and_save_nothing(interview, kwargs, fragment):
    events = await collect(interview.start(**kwargs))
    assert kinds(events) == ["Error"]
    assert events[0].fatal and fragment in events[0].content
    assert await database.list_interviews() == []


async def test_unknown_interview_type_falls_back_to_mixed(interview):
    await collect(interview.start("SWE", interview_type="nonsense"))
    assert (await database.get_interview(interview.id))["interview_type"] == "mixed"


async def test_bad_answers_are_not_fatal_and_keep_waiting(interview):
    await collect(interview.start("SWE"))
    for text, fragment in [("   ", "provide an answer"), ("x" * 5001, "too long")]:
        events = await collect(interview.answer(text))
        assert kinds(events) == ["Error"]
        assert not events[0].fatal and fragment in events[0].content
        assert interview.awaiting_input
    assert interview.questions_asked == 0


async def test_answer_before_start_is_rejected(interview):
    events = await collect(interview.answer("hello"))
    assert kinds(events) == ["Error"]


async def test_graph_failure_is_fatal_and_marks_interview_abandoned(interview):
    async def boom(messages, temperature=0.7):
        raise RuntimeError("provider down")

    with patch("app.agents.llm_service.complete", boom):
        events = await collect(interview.start("SWE"))
    assert kinds(events)[-1] == "Error"
    assert events[-1].fatal
    assert "provider down" not in events[-1].content
    assert (await database.get_interview(interview.id))["status"] == "abandoned"
    assert interview.status == "failed"


# --- Lifecycle: detached turns, resume, takeover ---

async def test_turn_survives_a_dropped_client_and_resume_replays_it(interview):
    await collect(interview.start("SWE"))

    events = interview.answer("My answer")
    assert type(await events.__anext__()).__name__ == "Typing"
    await events.aclose()  # client drops mid-turn
    interview.detach()
    await interview.idle()

    saved = await database.get_interview(interview.id)
    assert len(saved["messages"]) == 4  # the turn finished and was saved anyway

    resumed = await collect(interview.resume())
    assert kinds(resumed) == ["Resumed", "WaitingForInput"]
    history = resumed[0].messages
    assert [(m["type"], m.get("source")) for m in history] == [
        ("agent", "interviewer"), ("user", None), ("agent", "evaluator"), ("agent", "interviewer"),
    ]
    assert history[2]["metadata"]["score"] == 8
    assert resumed[0].questions_answered == 1


async def test_resume_during_a_running_turn_rejoins_without_duplicates(interview):
    await collect(interview.start("SWE"))
    gate = asyncio.Event()

    async def slow_complete(messages, temperature=0.7):
        await gate.wait()
        return await fake_complete(messages, temperature)

    with patch("app.agents.llm_service.complete", slow_complete):
        first = interview.answer("My answer")
        await first.__anext__()  # Typing; the turn is now blocked on the gate
        interview.detach()
        resumed = interview.resume()
        head = [await resumed.__anext__(), await resumed.__anext__()]
        assert kinds(head) == ["Resumed", "Typing"]
        assert head[0].messages == [{"type": "agent", "source": "interviewer",
                                     "content": "Describe a hard bug you fixed.", "metadata": {}}]
        gate.set()
        rest = [e async for e in resumed]
    assert kinds(rest) == ["AgentMessage", "AgentMessage", "WaitingForInput"]
    assert [e.source for e in rest[:2]] == ["evaluator", "interviewer"]


async def test_resume_takes_over_from_the_previous_client(interview):
    await collect(interview.start("SWE"))
    old = interview.resume()
    await old.__anext__()  # Resumed
    await old.__anext__()  # WaitingForInput
    new = await collect(interview.resume())
    assert kinds(new) == ["Resumed", "WaitingForInput"]


async def test_resume_after_completion_returns_the_scorecard(interview):
    await collect(interview.start("SWE"))
    await collect(interview.answer("a"))
    await collect(interview.end())
    resumed = await collect(interview.resume())
    assert kinds(resumed) == ["Resumed", "Summary", "Complete"]
    assert resumed[1].data["scores"]["technical"] == 7


async def test_abandon_marks_interview_abandoned(interview):
    await collect(interview.start("SWE"))
    await interview.abandon()
    assert interview.status == "abandoned"
    assert (await database.get_interview(interview.id))["status"] == "abandoned"
    assert kinds(await collect(interview.answer("late"))) == ["Error"]


async def test_checkpoint_cleanup_failure_does_not_lose_the_final_event():
    graph = build_interview_graph()
    interview = session.InterviewSession(graph)

    def explode(thread_id):
        raise RuntimeError("dictionary changed size during iteration")

    graph.checkpointer.delete_thread = explode
    await collect(interview.start("SWE"))
    await collect(interview.answer("a"))
    final = await collect(interview.end())
    assert kinds(final) == ["Typing", "Summary", "Complete"]
    assert interview.status == "completed"
