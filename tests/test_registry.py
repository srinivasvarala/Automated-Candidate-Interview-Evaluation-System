import asyncio
from unittest.mock import patch

import pytest
import pytest_asyncio

from app import database
from app.graph import build_interview_graph
from app.registry import SessionRegistry
from app.session import InterviewSession

pytestmark = pytest.mark.asyncio


async def _fake_complete(messages, temperature=0.7):
    return "A question?"


@pytest_asyncio.fixture
async def started():
    with patch("app.agents.llm_service.complete", _fake_complete):
        session = InterviewSession(build_interview_graph())
        _ = [e async for e in session.start("SWE")]
    return session


async def test_dropped_client_keeps_session_until_grace_expires(started):
    registry = SessionRegistry(grace_seconds=0.05)
    token = registry.add(started)
    registry.release(started, token)

    assert registry.get(started.id) is started
    await asyncio.sleep(0.15)

    assert registry.get(started.id) is None
    assert started.status == "abandoned"
    assert (await database.get_interview(started.id))["status"] == "abandoned"


async def test_reattaching_cancels_expiry(started):
    registry = SessionRegistry(grace_seconds=0.05)
    registry.release(started, registry.add(started))
    registry.claim(started)
    await asyncio.sleep(0.15)

    assert registry.get(started.id) is started
    assert started.status == "in_progress"


async def test_stale_release_does_not_detach_the_newer_client(started):
    registry = SessionRegistry(grace_seconds=0.05)
    old = registry.add(started)
    new = registry.claim(started)

    assert not registry.owns(started, old)
    registry.release(started, old)  # old socket closing late: ignored
    await asyncio.sleep(0.15)

    assert registry.owns(started, new)
    assert started.status == "in_progress"


async def test_finished_session_is_dropped_immediately():
    registry = SessionRegistry(grace_seconds=60)
    session = InterviewSession(build_interview_graph())  # never started
    registry.release(session, registry.add(session))
    assert registry.get(session.id) is None
