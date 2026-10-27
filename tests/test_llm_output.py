from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from app import llm_service
from app.llm_output import AnswerScore, ScorecardOutput


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))])


async def ask(*replies, model=AnswerScore):
    mock = AsyncMock(side_effect=[_response(r) for r in replies])
    with patch("app.llm_service.litellm.acompletion", mock):
        result = await llm_service.complete_model([{"role": "user", "content": "rate"}], model)
    return result, mock


@pytest.mark.asyncio
async def test_valid_reply():
    result, mock = await ask('{"score": 7}')
    assert result.score == 7
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_markdown_fenced_reply():
    result, _ = await ask('```json\n{"score": 9}\n```')
    assert result.score == 9


@pytest.mark.asyncio
async def test_retries_once_then_succeeds():
    result, mock = await ask("not json", '{"score": 6}')
    assert result.score == 6
    assert mock.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["not json", '{"score": 11}', '{"score": 0}', '{"other": 5}', "[]"])
async def test_gives_up_after_one_retry(bad):
    with pytest.raises(llm_service.LLMOutputError):
        await ask(bad, bad)


@pytest.mark.asyncio
async def test_empty_reply_is_an_error():
    with pytest.raises(llm_service.LLMOutputError):
        await ask("", "")


def test_scorecard_drops_invalid_categories_and_keeps_the_rest():
    card = ScorecardOutput.model_validate({
        "scores": {"technical": 8, "communication": 11, "culture_fit": "x", "problem_solving": 5},
        "strengths": ["a"],
    })
    assert card.scores == {"technical": 8, "problem_solving": 5}
    assert card.improvements == []
    assert card.overall_summary == ""


def test_scorecard_needs_at_least_one_valid_score():
    with pytest.raises(ValidationError):
        ScorecardOutput.model_validate({"scores": {"technical": 99}})


def test_scorecard_resume_analysis_is_optional():
    card = ScorecardOutput.model_validate({
        "scores": {"technical": 8},
        "resume_analysis": {"verified_skills": ["Python"]},
    })
    assert card.model_dump(exclude_none=True)["resume_analysis"]["gaps"] == []
    assert "resume_analysis" not in ScorecardOutput.model_validate({"scores": {"technical": 8}}).model_dump(exclude_none=True)
