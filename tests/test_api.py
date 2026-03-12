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
