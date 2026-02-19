from __future__ import annotations

from typing import Any


# In-memory session store keyed by session_id
_sessions: dict[str, dict[str, Any]] = {}


def create_session(session_id: str, job_position: str) -> dict[str, Any]:
    _sessions[session_id] = {
        "job_position": job_position,
        "status": "active",
    }
    return _sessions[session_id]


def get_session(session_id: str) -> dict[str, Any] | None:
    return _sessions.get(session_id)


def remove_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
