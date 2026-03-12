import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# Set test environment variables before importing app modules
os.environ["LLM_API_KEY"] = "test-key-for-testing"
os.environ["LLM_MODEL"] = "openai/gpt-4o"
os.environ["NUM_QUESTIONS"] = "3"

from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture
def sample_state():
    """Return a minimal GraphState for testing."""
    return {
        "messages": [HumanMessage(content="Start the interview for a Software Engineer position.")],
        "job_position": "Software Engineer",
        "job_description": "",
        "interview_type": "mixed",
        "questions_asked": 0,
        "current_phase": "interviewer",
        "end_requested": False,
        "summary_data": None,
    }


@pytest.fixture
def sample_state_with_jd():
    """Return a GraphState with a job description."""
    return {
        "messages": [HumanMessage(content="Start the interview for an AI Engineer position.")],
        "job_position": "AI Engineer",
        "job_description": "We are looking for an AI Engineer with experience in LLMs, RAG pipelines, and vector databases.",
        "interview_type": "technical",
        "questions_asked": 0,
        "current_phase": "interviewer",
        "end_requested": False,
        "summary_data": None,
    }


@pytest.fixture
def state_with_conversation():
    """Return a GraphState with conversation history."""
    return {
        "messages": [
            HumanMessage(content="Start the interview for a Software Engineer position."),
            AIMessage(content="Tell me about a challenging project you worked on.", name="interviewer"),
            HumanMessage(content="I built a distributed caching system that reduced latency by 40%.", name="candidate"),
        ],
        "job_position": "Software Engineer",
        "job_description": "",
        "interview_type": "mixed",
        "questions_asked": 1,
        "current_phase": "evaluator",
        "end_requested": False,
        "summary_data": None,
    }


@pytest.fixture
def mock_llm_response():
    """Return a mock for llm_service.complete that returns a string."""
    mock = AsyncMock(return_value="This is a mock LLM response.")
    return mock


@pytest.fixture
def mock_llm_json_response():
    """Return a mock for llm_service.complete_json that returns a JSON string."""
    mock = AsyncMock(return_value='{"scores": {"technical": 7, "problem_solving": 8, "communication": 6, "culture_fit": 7}, "strengths": ["Strong technical depth", "Clear communication", "Good problem-solving approach"], "improvements": ["Could provide more specific examples", "Consider edge cases", "Structure answers using STAR method"], "overall_summary": "The candidate demonstrated solid technical skills and good communication. With more structured responses, they could improve significantly."}')
    return mock


@pytest.fixture
def mock_llm_score_response():
    """Return a mock for llm_service.complete_json that returns a score."""
    mock = AsyncMock(return_value='{"score": 7}')
    return mock
