from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from app import llm_service

pytestmark = pytest.mark.asyncio


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))])


def _temperature_rejected():
    return litellm.BadRequestError(
        message='{"message":"`temperature` is deprecated for this model."}',
        model="anthropic/new-model",
        llm_provider="anthropic",
    )


@pytest.fixture(autouse=True)
def reset_learned_models():
    llm_service._NO_TEMPERATURE.clear()
    yield
    llm_service._NO_TEMPERATURE.clear()


async def test_retries_without_temperature_when_model_rejects_it():
    mock = AsyncMock(side_effect=[_temperature_rejected(), _response("hello")])
    with patch("app.llm_service.litellm.acompletion", mock):
        assert await llm_service.complete([{"role": "user", "content": "hi"}]) == "hello"

    assert "temperature" in mock.call_args_list[0].kwargs
    assert "temperature" not in mock.call_args_list[1].kwargs


async def test_remembers_the_model_and_skips_temperature_next_time():
    mock = AsyncMock(side_effect=[_temperature_rejected(), _response("one"), _response("two")])
    with patch("app.llm_service.litellm.acompletion", mock):
        await llm_service.complete([{"role": "user", "content": "hi"}])
        await llm_service.complete([{"role": "user", "content": "again"}])

    assert mock.call_count == 3  # the second request needed no retry
    assert "temperature" not in mock.call_args_list[2].kwargs


async def test_json_calls_get_the_same_treatment():
    mock = AsyncMock(side_effect=[_temperature_rejected(), _response('{"score": 7}')])
    with patch("app.llm_service.litellm.acompletion", mock):
        assert await llm_service.complete_json([{"role": "user", "content": "rate"}]) == '{"score": 7}'


async def test_other_bad_requests_are_not_swallowed():
    other = litellm.BadRequestError(message="prompt is too long", model="m", llm_provider="anthropic")
    mock = AsyncMock(side_effect=other)
    with patch("app.llm_service.litellm.acompletion", mock), pytest.raises(litellm.BadRequestError):
        await llm_service.complete([{"role": "user", "content": "hi"}])
    assert mock.call_count == 1
