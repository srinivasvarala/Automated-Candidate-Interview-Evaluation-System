import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app import database, prep
from app.main import app

LESSON = {
    "case_title": "RAG support bot",
    "scenario": "10,000 documents.",
    "design": "1. Ingest\n2. Retrieve",
    "trade_offs": "Hybrid vs dense.",
    "failure_modes": "Stale index.",
    "testing_and_observability": "Eval set.",
    "interview_angle": "Lead with requirements.",
    "checks": ["How would you evaluate it?", "What breaks at 10x?"],
}


def outline_json():
    return json.dumps({"areas": [
        {"key": key, "title": title, "case": f"Case for {key}", "goal": f"Goal for {key}"}
        for key, title in prep.BACKBONE
    ]})


def llm(complete_json_replies, feedback="Good answer."):
    """Patch the LLM: JSON replies are consumed in order, free text is fixed."""
    mock = AsyncMock(side_effect=complete_json_replies) if isinstance(complete_json_replies, list) \
        else AsyncMock(return_value=complete_json_replies)
    return mock, patch("app.llm_service.complete_json", mock), patch("app.llm_service.complete", AsyncMock(return_value=feedback))


@pytest.fixture
def client():
    return TestClient(app)


def create_plan(client, **body):
    body = {"role": "AI Engineer", "seniority": "senior", "job_description": "jd", "resume_text": "my secret cv", **body}
    with patch("app.llm_service.complete_json", AsyncMock(return_value=outline_json())):
        return client.post("/api/prep/plans", json=body)


# --- templates ---

def test_templates(client):
    data = client.get("/api/prep/templates").json()
    assert data["seniorities"] == ["junior", "mid", "senior", "staff"]
    assert data["default_seniority"] == "mid"
    assert "AI Architect" in [t["role"] for t in data["templates"]]


# --- create ---

def test_create_plan(client):
    response = create_plan(client)
    assert response.status_code == 201
    plan = response.json()
    assert (plan["role"], plan["seniority"]) == ("AI Engineer", "senior")
    assert [a["key"] for a in plan["areas"]] == [k for k, _ in prep.BACKBONE]
    assert all(a["status"] == "not_started" and a["lesson"] is None for a in plan["areas"])
    assert plan["has_resume"] is True
    assert "resume_text" not in plan and "my secret cv" not in response.text


def test_create_defaults_seniority_to_mid(client):
    assert create_plan(client, seniority="").json()["seniority"] == "mid"


@pytest.mark.parametrize(
    "body, message",
    [
        ({"role": ""}, "required"),
        ({"role": "x" * 201}, "200"),
        ({"seniority": "wizard"}, "Seniority"),
        ({"job_description": "x" * 10_001}, "10,000"),
        ({"role": 5}, "text"),
    ],
)
def test_create_rejects_bad_input_without_calling_the_model(client, body, message):
    mock = AsyncMock()
    with patch("app.llm_service.complete_json", mock):
        response = client.post("/api/prep/plans", json={"role": "SRE", **body})
    assert response.status_code == 400
    assert message in response.json()["error"]
    mock.assert_not_called()


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", ""])
def test_create_rejects_a_body_that_is_not_a_json_object(client, raw):
    response = client.post("/api/prep/plans", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 400


def test_create_with_unusable_model_output_returns_502_and_saves_nothing(client):
    mock = AsyncMock(return_value="garbage")
    with patch("app.llm_service.complete_json", mock):
        response = client.post("/api/prep/plans", json={"role": "SRE"})
    assert response.status_code == 502
    assert "try again" in response.json()["error"]
    assert mock.call_count == 2  # one retry
    assert client.get("/api/prep/plans").json() == {"plans": [], "count": 0}


def test_create_when_the_model_call_itself_fails_returns_502(client):
    with patch("app.llm_service.complete_json", AsyncMock(side_effect=RuntimeError("provider down"))):
        response = client.post("/api/prep/plans", json={"role": "SRE"})
    assert response.status_code == 502
    assert "provider down" not in response.text


# --- list and get ---

def test_list_and_get(client):
    plan_id = create_plan(client).json()["id"]
    listed = client.get("/api/prep/plans").json()
    assert listed["count"] == 1
    item = listed["plans"][0]
    assert (item["id"], item["areas_total"], item["lessons_generated"]) == (plan_id, 10, 0)
    assert "resume_text" not in item
    assert client.get(f"/api/prep/plans/{plan_id}").json()["id"] == plan_id


def test_get_unknown_plan_is_404(client):
    response = client.get("/api/prep/plans/nope")
    assert response.status_code == 404
    assert response.json()["error"] == "Prep plan not found"


def test_list_validates_paging(client):
    assert client.get("/api/prep/plans?limit=0").status_code == 422


# --- lessons ---

def open_lesson(client, plan_id, area="architecture", reply=None):
    mock, a, b = llm(json.dumps(reply or LESSON))
    with a, b:
        return client.post(f"/api/prep/plans/{plan_id}/lessons/{area}"), mock


def test_open_lesson_generates_then_serves_it_from_storage(client):
    plan_id = create_plan(client).json()["id"]
    response, mock = open_lesson(client, plan_id)
    assert response.status_code == 200
    area = response.json()
    assert (area["key"], area["status"], area["lesson"]["case_title"]) == ("architecture", "ready", "RAG support bot")
    assert mock.call_count == 1

    again, mock2 = open_lesson(client, plan_id)
    assert again.json()["lesson"] == area["lesson"]
    mock2.assert_not_called()
    assert client.get(f"/api/prep/plans/{plan_id}").json()["areas"][0]["status"] == "ready"
    assert client.get("/api/prep/plans").json()["plans"][0]["lessons_generated"] == 1


def test_open_lesson_unknown_plan_or_area_is_404(client):
    plan_id = create_plan(client).json()["id"]
    assert open_lesson(client, "nope")[0].status_code == 404
    response, mock = open_lesson(client, plan_id, area="astrology")
    assert response.status_code == 404
    mock.assert_not_called()


def test_failed_lesson_returns_502_and_saves_nothing(client):
    plan_id = create_plan(client).json()["id"]
    response, mock = open_lesson(client, plan_id, reply={**LESSON, "checks": ["only one"]})
    assert response.status_code == 502
    assert mock.call_count == 2
    assert client.get(f"/api/prep/plans/{plan_id}").json()["areas"][0]["status"] == "not_started"
    # and it can simply be tried again
    assert open_lesson(client, plan_id)[0].status_code == 200


@pytest.mark.asyncio
async def test_simultaneous_requests_generate_a_lesson_once():
    calls = 0

    async def slow_json(messages, temperature=0.3, timeout=60):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return json.dumps(LESSON)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        with patch("app.llm_service.complete_json", AsyncMock(return_value=outline_json())):
            plan_id = (await client.post("/api/prep/plans", json={"role": "SRE"})).json()["id"]
        with patch("app.llm_service.complete_json", slow_json):
            first, second = await asyncio.gather(
                client.post(f"/api/prep/plans/{plan_id}/lessons/design"),
                client.post(f"/api/prep/plans/{plan_id}/lessons/design"),
            )
    assert (first.status_code, second.status_code) == (200, 200)
    assert first.json()["lesson"] == second.json()["lesson"]
    assert calls == 1


# --- checks ---

def answer(client, plan_id, idx=0, text="Golden set plus online metrics.", area="architecture", score='{"score": 8}'):
    with patch("app.llm_service.complete", AsyncMock(return_value="Good. Add failure modes.")), \
         patch("app.llm_service.complete_json", AsyncMock(return_value=score)):
        return client.post(f"/api/prep/plans/{plan_id}/lessons/{area}/checks/{idx}", json={"answer": text})


@pytest.fixture
def plan_with_lesson(client):
    plan_id = create_plan(client).json()["id"]
    open_lesson(client, plan_id)
    return plan_id


def test_answering_a_check_returns_feedback_and_a_score_and_saves_it(client, plan_with_lesson):
    response = answer(client, plan_with_lesson)
    assert response.status_code == 200
    assert response.json() == {
        "idx": 0, "question": "How would you evaluate it?", "answer": "Golden set plus online metrics.",
        "score": 8, "feedback": "Good. Add failure modes.",
    }
    checks = client.get(f"/api/prep/plans/{plan_with_lesson}").json()["areas"][0]["checks"]
    assert [(c["idx"], c["score"]) for c in checks] == [(0, 8)]


def test_an_unscorable_check_still_returns_feedback(client, plan_with_lesson):
    body = answer(client, plan_with_lesson, score="garbage").json()
    assert body["score"] is None and body["feedback"] == "Good. Add failure modes."


def test_answering_again_replaces_the_earlier_answer(client, plan_with_lesson):
    answer(client, plan_with_lesson, text="first", score='{"score": 3}')
    answer(client, plan_with_lesson, text="second", score='{"score": 9}')
    checks = client.get(f"/api/prep/plans/{plan_with_lesson}").json()["areas"][0]["checks"]
    assert [(c["answer"], c["score"]) for c in checks] == [("second", 9)]


@pytest.mark.parametrize("payload", [{}, {"answer": ""}, {"answer": "   "}, {"answer": 5}, {"answer": None}])
def test_empty_or_wrong_type_answers_are_400(client, plan_with_lesson, payload):
    response = client.post(f"/api/prep/plans/{plan_with_lesson}/lessons/architecture/checks/0", json=payload)
    assert response.status_code == 400


def test_too_long_answer_is_400(client, plan_with_lesson):
    assert answer(client, plan_with_lesson, text="x" * 5001).status_code == 400


def test_check_before_the_lesson_exists_is_409(client):
    plan_id = create_plan(client).json()["id"]
    assert answer(client, plan_id).status_code == 409


@pytest.mark.parametrize("idx", ["2", "-1", "abc", "99"])
def test_unknown_check_is_404(client, plan_with_lesson, idx):
    assert answer(client, plan_with_lesson, idx=idx).status_code == 404


def test_check_for_unknown_plan_or_area_is_404(client, plan_with_lesson):
    assert answer(client, "nope").status_code == 404
    assert answer(client, plan_with_lesson, area="astrology").status_code == 404


def test_feedback_failure_is_502_and_saves_nothing(client, plan_with_lesson):
    with patch("app.llm_service.complete", AsyncMock(side_effect=RuntimeError("down"))):
        response = client.post(
            f"/api/prep/plans/{plan_with_lesson}/lessons/architecture/checks/0", json={"answer": "x"})
    assert response.status_code == 502
    assert client.get(f"/api/prep/plans/{plan_with_lesson}").json()["areas"][0]["checks"] == []


# --- complete ---

def test_complete_a_lesson(client, plan_with_lesson):
    response = client.post(f"/api/prep/plans/{plan_with_lesson}/lessons/architecture/complete")
    assert response.status_code == 200
    assert response.json()["status"] == "done"
    assert client.get("/api/prep/plans").json()["plans"][0]["lessons_done"] == 1


def test_complete_before_the_lesson_exists_is_409(client):
    plan_id = create_plan(client).json()["id"]
    assert client.post(f"/api/prep/plans/{plan_id}/lessons/architecture/complete").status_code == 409


def test_complete_unknown_plan_or_area_is_404(client, plan_with_lesson):
    assert client.post("/api/prep/plans/nope/lessons/architecture/complete").status_code == 404
    assert client.post(f"/api/prep/plans/{plan_with_lesson}/lessons/astrology/complete").status_code == 404


def test_nothing_leaks_to_the_database_on_validation_errors(client):
    client.post("/api/prep/plans", json={"role": ""})
    assert asyncio.run(database.list_prep_plans()) == []
