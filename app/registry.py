"""Live Interview sessions, kept briefly after their client disconnects.

A client that drops (Wi-Fi, refresh) can re-attach within the grace period.
After that the session is abandoned. State is in memory, so nothing survives
a server restart.
"""

import asyncio
import logging

from app.session import InterviewSession

logger = logging.getLogger(__name__)

DEFAULT_GRACE_SECONDS = 15 * 60


class SessionRegistry:
    def __init__(self, grace_seconds: float = DEFAULT_GRACE_SECONDS):
        self._grace = grace_seconds
        self._sessions: dict[str, InterviewSession] = {}
        self._owners: dict[str, object] = {}
        self._timers: dict[str, asyncio.Task] = {}

    def get(self, session_id: str) -> InterviewSession | None:
        return self._sessions.get(session_id)

    def add(self, session: InterviewSession) -> object:
        """Register a new session. Returns the connection token for its first client."""
        self._sessions[session.id] = session
        return self.claim(session)

    def claim(self, session: InterviewSession) -> object:
        """A client attaches. Any previous client loses ownership."""
        self._cancel_timer(session.id)
        token = object()
        self._owners[session.id] = token
        return token

    def owns(self, session: InterviewSession, token: object) -> bool:
        return self._owners.get(session.id) is token

    def release(self, session: InterviewSession, token: object) -> None:
        """A client went away. No-op if it was already replaced by a newer one."""
        if not self.owns(session, token):
            return
        del self._owners[session.id]
        session.detach()
        if session.status == "in_progress":
            self._timers[session.id] = asyncio.create_task(self._expire(session))
        else:
            self._sessions.pop(session.id, None)

    async def _expire(self, session: InterviewSession) -> None:
        await asyncio.sleep(self._grace)
        logger.info("[%s] Grace period over, abandoning", session.id)
        self._timers.pop(session.id, None)
        self._sessions.pop(session.id, None)
        await session.abandon()

    def _cancel_timer(self, session_id: str) -> None:
        timer = self._timers.pop(session_id, None)
        if timer is not None:
            timer.cancel()
