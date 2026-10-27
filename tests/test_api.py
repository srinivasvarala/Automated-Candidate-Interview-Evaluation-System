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
        assert "AI Interview Coach" in response.text


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
