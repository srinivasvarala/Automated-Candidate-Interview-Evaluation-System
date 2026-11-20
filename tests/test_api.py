import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model" in data

    def test_health_returns_model(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["model"] == "openai/gpt-4o"


class TestIndexEndpoint:
    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Agentic Interview Coach" in response.text


class TestErrorStatusCodes:
    def test_unsupported_resume_type_returns_400(self, client):
        response = client.post("/api/upload-resume", files={"file": ("cv.docx", b"x")})
        assert response.status_code == 400
        assert "Unsupported" in response.json()["error"]

    def test_oversized_resume_returns_400(self, client):
        big = b"a" * (5 * 1024 * 1024 + 1)
        response = client.post("/api/upload-resume", files={"file": ("cv.txt", big)})
        assert response.status_code == 400

    def test_unknown_interview_returns_404(self, client):
        with patch("app.main.database.get_interview", new=AsyncMock(return_value=None)):
            response = client.get("/api/interviews/nope")
        assert response.status_code == 404
        assert response.json()["error"] == "Interview not found"


class TestWebSocketInterview:
    def test_rejects_empty_job_position(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            ws.send_text(json.dumps({"job_position": ""}))
            data = json.loads(ws.receive_text())
            assert data["type"] == "error"
            assert "required" in data["content"].lower()

    def test_rejects_long_job_position(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            ws.send_text(json.dumps({"job_position": "x" * 201}))
            data = json.loads(ws.receive_text())
            assert data["type"] == "error"
            assert "200" in data["content"]

    def test_rejects_long_job_description(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            ws.send_text(json.dumps({
                "job_position": "Engineer",
                "job_description": "x" * 10001,
            }))
            data = json.loads(ws.receive_text())
            assert data["type"] == "error"
            assert "10,000" in data["content"]

    def test_accepts_valid_start_message(self, client):
        """Test that a valid start message triggers interview_started event."""
        mock_stream = AsyncMock()
        mock_stream.__aiter__ = MagicMock(return_value=iter([]))

        with patch("app.main.interview_graph.astream", return_value=mock_stream):
            with client.websocket_connect("/ws/interview") as ws:
                ws.send_text(json.dumps({
                    "job_position": "Software Engineer",
                    "interview_type": "technical",
                }))
                data = json.loads(ws.receive_text())
                assert data["type"] == "system_event"
                assert data["event"] == "interview_started"
                assert data["metadata"]["num_questions"] > 0

    def test_full_interview_wire_format(self, client):
        async def complete(messages, temperature=0.7):
            return "Feedback. 💡 **Stronger answer:** More detail." if "career coach" in messages[0]["content"] else "A question?"

        async def complete_json(messages, temperature=0.3):
            if "Rate the candidate" in messages[0]["content"]:
                return '{"score": 9}'
            return '{"scores": {"technical": 9}, "strengths": [], "improvements": [], "overall_summary": "ok"}'

        def recv(ws):
            return json.loads(ws.receive_text())

        with patch("app.agents.llm_service.complete", complete), \
             patch("app.agents.llm_service.complete_json", complete_json), \
             client.websocket_connect("/ws/interview") as ws:
            ws.send_text(json.dumps({"job_position": "Engineer"}))
            assert recv(ws)["event"] == "interview_started"
            assert recv(ws)["event"] == "agent_typing"
            assert recv(ws)["source"] == "interviewer"
            assert recv(ws)["event"] == "waiting_for_input"

            ws.send_text(json.dumps({"type": "user_input", "content": "My answer"}))
            assert recv(ws)["event"] == "agent_typing"
            feedback = recv(ws)
            assert feedback["source"] == "evaluator"
            assert feedback["metadata"] == {"score": 9, "tier": "Warm-up", "tier_index": 0}
            assert recv(ws)["source"] == "interviewer"
            assert recv(ws)["event"] == "waiting_for_input"

            ws.send_text(json.dumps({"type": "end_interview"}))
            assert recv(ws)["event"] == "agent_typing"
            summary = recv(ws)
            assert summary["type"] == "summary"
            assert summary["metadata"]["scores"] == {"technical": 9}
            assert recv(ws)["event"] == "interview_complete"

    def test_resume_after_disconnect_replays_the_interview(self):
        async def complete(messages, temperature=0.7):
            return "Feedback." if "career coach" in messages[0]["content"] else "A question?"

        async def complete_json(messages, temperature=0.3):
            return '{"score": 6}'

        def recv(ws):
            return json.loads(ws.receive_text())

        with patch("app.agents.llm_service.complete", complete), \
             patch("app.agents.llm_service.complete_json", complete_json), \
             TestClient(app) as client:
            with client.websocket_connect("/ws/interview") as ws:
                ws.send_text(json.dumps({"job_position": "Engineer"}))
                session_id = recv(ws)["metadata"]["session_id"]
                while recv(ws).get("event") != "waiting_for_input":
                    pass
            # socket closed: the session stays alive for resume

            with client.websocket_connect("/ws/interview") as ws:
                ws.send_text(json.dumps({"type": "resume", "session_id": session_id}))
                resumed = recv(ws)
                assert resumed["event"] == "resumed"
                assert [m["source"] for m in resumed["metadata"]["messages"]] == ["interviewer"]
                assert recv(ws)["event"] == "waiting_for_input"

                ws.send_text(json.dumps({"type": "user_input", "content": "Answer after resume"}))
                assert recv(ws)["event"] == "agent_typing"
                assert recv(ws)["source"] == "evaluator"

    def test_resume_unknown_session_reports_expired(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            ws.send_text(json.dumps({"type": "resume", "session_id": "nope"}))
            data = json.loads(ws.receive_text())
            assert data["type"] == "error"
            assert data["code"] == "session_not_found"


class TestMockFromPrepPlan:
    """A mock interview started from a Prep plan: the server adds the plan's resume and focus."""

    @staticmethod
    def make_plan(plan_id="plan-1"):
        import asyncio

        from app import database

        outline = [
            {"key": "architecture", "title": "Architecture", "case": "RAG bot", "goal": "g"},
            {"key": "observability", "title": "Observability", "case": "Tracing", "goal": "g"},
        ]

        async def create():
            await database.create_prep_plan(plan_id, "AI Architect", "staff", "Own the AI platform", "Led a RAG rollout", outline)
            await database.save_lesson(plan_id, "observability", {"case_title": "t", "checks": ["q1", "q2"]})
            await database.save_check(plan_id, "observability", 0, "q1", "a", 2, "fb")
            await database.save_check(plan_id, "architecture", 0, "q1", "a", 9, "fb")

        asyncio.run(create())

    def test_plan_fills_in_the_resume_and_focus_the_client_never_sees(self, client):
        self.make_plan()
        prompts = []

        async def spy_complete(messages, temperature=0.7):
            prompts.append(messages[0]["content"])
            return "A question?"

        with patch("app.agents.llm_service.complete", spy_complete), \
             client.websocket_connect("/ws/interview") as ws:
            ws.send_text(json.dumps({
                "job_position": "AI Architect", "interview_type": "technical", "prep_plan_id": "plan-1",
            }))
            assert json.loads(ws.receive_text())["event"] == "interview_started"
            while json.loads(ws.receive_text()).get("event") != "waiting_for_input":
                pass

        system = prompts[0]
        assert "Led a RAG rollout" in system  # resume from the plan
        assert "Own the AI platform" in system  # job description from the plan
        assert "## Prep Focus" in system
        assert system.index("- Observability") < system.index("- Architecture")  # weakest first

    def test_client_values_win_over_the_plan(self, client):
        self.make_plan()
        prompts = []

        async def spy_complete(messages, temperature=0.7):
            prompts.append(messages[0]["content"])
            return "A question?"

        with patch("app.agents.llm_service.complete", spy_complete), \
             client.websocket_connect("/ws/interview") as ws:
            ws.send_text(json.dumps({
                "job_position": "Other Role", "job_description": "My own JD", "prep_plan_id": "plan-1",
            }))
            while json.loads(ws.receive_text()).get("event") != "waiting_for_input":
                pass

        assert "Other Role" in prompts[0] and "My own JD" in prompts[0]
        assert "Own the AI platform" not in prompts[0]

    def test_blank_role_falls_back_to_the_plan_role(self, client):
        self.make_plan()
        prompts = []

        async def spy_complete(messages, temperature=0.7):
            prompts.append(messages[0]["content"])
            return "A question?"

        with patch("app.agents.llm_service.complete", spy_complete), \
             client.websocket_connect("/ws/interview") as ws:
            ws.send_text(json.dumps({"prep_plan_id": "plan-1"}))
            while json.loads(ws.receive_text()).get("event") != "waiting_for_input":
                pass

        assert "AI Architect" in prompts[0]

    def test_unknown_plan_is_a_fatal_error_and_starts_nothing(self, client):
        started = []
        with patch("app.main.session.InterviewSession.start", lambda *a, **k: started.append(1)), \
             client.websocket_connect("/ws/interview") as ws:
            ws.send_text(json.dumps({"job_position": "Engineer", "prep_plan_id": "nope"}))
            data = json.loads(ws.receive_text())
        assert (data["type"], data["code"], data["fatal"]) == ("error", "prep_plan_not_found", True)
        assert not started

    def test_without_a_plan_nothing_changes(self, client):
        prompts = []

        async def spy_complete(messages, temperature=0.7):
            prompts.append(messages[0]["content"])
            return "A question?"

        with patch("app.agents.llm_service.complete", spy_complete), \
             client.websocket_connect("/ws/interview") as ws:
            ws.send_text(json.dumps({"job_position": "Engineer"}))
            while json.loads(ws.receive_text()).get("event") != "waiting_for_input":
                pass
        assert "Prep Focus" not in prompts[0]
