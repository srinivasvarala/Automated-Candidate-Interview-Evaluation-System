from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents import (
    interviewer_node,
    evaluator_node,
    summary_node,
    _build_messages,
    _compute_adaptive_tier,
    _update_tracker,
    _init_tracker,
    _clean_evaluator_response,
)


class TestBuildMessages:
    def test_converts_human_messages(self, sample_state):
        msgs = _build_messages(sample_state, "You are an interviewer.")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_converts_ai_messages(self, state_with_conversation):
        msgs = _build_messages(state_with_conversation, "System prompt")
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]

    def test_system_prompt_is_first(self, sample_state):
        prompt = "Custom system prompt"
        msgs = _build_messages(sample_state, prompt)
        assert msgs[0] == {"role": "system", "content": prompt}


class TestInterviewerNode:
    @pytest.mark.asyncio
    async def test_returns_ai_message(self, sample_state, mock_llm_response):
        with patch("app.agents.llm_service.complete", mock_llm_response):
            result = await interviewer_node(sample_state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].name == "interviewer"

    @pytest.mark.asyncio
    async def test_sets_phase_to_candidate(self, sample_state, mock_llm_response):
        with patch("app.agents.llm_service.complete", mock_llm_response):
            result = await interviewer_node(sample_state)

        assert result["current_phase"] == "candidate"

    @pytest.mark.asyncio
    async def test_includes_jd_in_prompt(self, sample_state_with_jd, mock_llm_response):
        with patch("app.agents.llm_service.complete", mock_llm_response) as mock:
            await interviewer_node(sample_state_with_jd)

        # Check that the system prompt includes the JD
        call_args = mock.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "LLMs" in system_msg or "job description" in system_msg.lower()


class TestEvaluatorNode:
    @pytest.mark.asyncio
    async def test_returns_ai_message(self, state_with_conversation, mock_llm_response, mock_llm_score_response):
        with patch("app.agents.llm_service.complete", mock_llm_response), \
             patch("app.agents.llm_service.complete_json", mock_llm_score_response):
            result = await evaluator_node(state_with_conversation)

        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].name == "evaluator"

    @pytest.mark.asyncio
    async def test_increments_questions_asked(self, state_with_conversation, mock_llm_response, mock_llm_score_response):
        with patch("app.agents.llm_service.complete", mock_llm_response), \
             patch("app.agents.llm_service.complete_json", mock_llm_score_response):
            result = await evaluator_node(state_with_conversation)

        assert result["questions_asked"] == 2  # was 1, now 2

    @pytest.mark.asyncio
    async def test_sets_phase_to_interviewer(self, state_with_conversation, mock_llm_response, mock_llm_score_response):
        with patch("app.agents.llm_service.complete", mock_llm_response), \
             patch("app.agents.llm_service.complete_json", mock_llm_score_response):
            result = await evaluator_node(state_with_conversation)

        assert result["current_phase"] == "interviewer"

    @pytest.mark.asyncio
    async def test_updates_performance_tracker(self, state_with_conversation, mock_llm_response, mock_llm_score_response):
        with patch("app.agents.llm_service.complete", mock_llm_response), \
             patch("app.agents.llm_service.complete_json", mock_llm_score_response):
            result = await evaluator_node(state_with_conversation)

        tracker = result["performance_tracker"]
        assert tracker is not None
        assert tracker["scores"] == [7]
        assert len(tracker["tiers"]) == 1

    @pytest.mark.asyncio
    async def test_handles_bad_score_json(self, state_with_conversation, mock_llm_response):
        bad_score = AsyncMock(return_value="not json")
        with patch("app.agents.llm_service.complete", mock_llm_response), \
             patch("app.agents.llm_service.complete_json", bad_score):
            result = await evaluator_node(state_with_conversation)

        # Should default to score 5
        assert result["performance_tracker"]["scores"] == [5]


class TestSummaryNode:
    @pytest.mark.asyncio
    async def test_parses_valid_json(self, state_with_conversation, mock_llm_json_response):
        with patch("app.agents.llm_service.complete_json", mock_llm_json_response):
            result = await summary_node(state_with_conversation)

        assert result["summary_data"]["scores"]["technical"] == 7
        assert len(result["summary_data"]["strengths"]) == 3
        assert result["current_phase"] == "done"

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self, state_with_conversation):
        bad_mock = AsyncMock(return_value="This is not JSON at all")
        with patch("app.agents.llm_service.complete_json", bad_mock):
            result = await summary_node(state_with_conversation)

        assert result["summary_data"]["scores"]["technical"] == 5
        assert "Unable to parse" in result["summary_data"]["strengths"][0]

    @pytest.mark.asyncio
    async def test_returns_summary_ai_message(self, state_with_conversation, mock_llm_json_response):
        with patch("app.agents.llm_service.complete_json", mock_llm_json_response):
            result = await summary_node(state_with_conversation)

        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].name == "summary"


class TestAdaptiveDifficulty:
    def test_init_tracker(self):
        tracker = _init_tracker()
        assert tracker["scores"] == []
        assert tracker["tiers"] == []
        assert tracker["current_tier"] == 0
        assert tracker["consecutive_strong"] == 0
        assert tracker["consecutive_weak"] == 0

    def test_default_tier_for_early_questions(self):
        tracker = _init_tracker()
        assert _compute_adaptive_tier(tracker, 1) == 0  # warm-up
        assert _compute_adaptive_tier(tracker, 2) == 1  # foundation

    def test_tier_up_after_two_strong(self):
        tracker = {"scores": [], "tiers": [], "current_tier": 2,
                    "consecutive_strong": 0, "consecutive_weak": 0}
        tracker = _update_tracker(tracker, 8, 2)  # strong
        tracker = _update_tracker(tracker, 7, 2)  # strong again → should tier up
        assert tracker["consecutive_strong"] == 2
        assert tracker["current_tier"] == 3  # went from 2 → 3

    def test_tier_down_after_two_weak(self):
        tracker = {"scores": [], "tiers": [], "current_tier": 3,
                    "consecutive_strong": 0, "consecutive_weak": 0}
        tracker = _update_tracker(tracker, 3, 3)  # weak
        tracker = _update_tracker(tracker, 4, 3)  # weak again → should tier down
        assert tracker["consecutive_weak"] == 2
        assert tracker["current_tier"] <= 2

    def test_tier_stays_on_mixed_scores(self):
        tracker = {"scores": [], "tiers": [], "current_tier": 2,
                    "consecutive_strong": 0, "consecutive_weak": 0}
        tracker = _update_tracker(tracker, 6, 2)  # mid
        assert tracker["consecutive_strong"] == 0
        assert tracker["consecutive_weak"] == 0
        assert tracker["current_tier"] == 2

    def test_tier_capped_at_max(self):
        tracker = {"scores": [], "tiers": [], "current_tier": 4,
                    "consecutive_strong": 2, "consecutive_weak": 0}
        tracker = _update_tracker(tracker, 9, 4)  # strong but already at max
        assert tracker["current_tier"] <= 4

    def test_tier_floored_at_zero(self):
        tracker = {"scores": [], "tiers": [], "current_tier": 0,
                    "consecutive_strong": 0, "consecutive_weak": 2}
        tracker = _update_tracker(tracker, 2, 0)  # weak but already at min
        assert tracker["current_tier"] >= 0

    def test_streak_resets_on_mid_score(self):
        tracker = _init_tracker()
        tracker = _update_tracker(tracker, 8, 0)  # strong
        assert tracker["consecutive_strong"] == 1
        tracker = _update_tracker(tracker, 5, 1)  # mid → resets streak
        assert tracker["consecutive_strong"] == 0
        assert tracker["consecutive_weak"] == 0


class TestCleanEvaluatorResponse:
    def test_keeps_clean_response(self):
        text = "Good answer with specifics.\n\n💡 **Stronger answer:** A better response would mention X and Y."
        assert _clean_evaluator_response(text) == text.rstrip()

    def test_strips_trailing_question_after_hr(self):
        text = (
            "Good answer.\n\n"
            "💡 **Stronger answer:** Example response here.\n\n"
            "---\n\n"
            "You're designing a system that handles millions of requests. How would you architect this?"
        )
        result = _clean_evaluator_response(text)
        assert "How would you architect" not in result
        assert "Stronger answer" in result

    def test_strips_trailing_question_without_hr(self):
        text = (
            "Good feedback here.\n\n"
            "💡 **Stronger answer:** Example response.\n\n"
            "Tell me about a time when you had to optimize a machine learning model in production. What was your approach?"
        )
        result = _clean_evaluator_response(text)
        assert "Tell me about" not in result
        assert "Stronger answer" in result

    def test_preserves_question_marks_in_quotes(self):
        text = (
            "Good answer.\n\n"
            '💡 **Stronger answer:** "What if we used caching here?" is the kind of question you should ask.'
        )
        result = _clean_evaluator_response(text)
        assert "What if we used caching" in result

    def test_handles_empty_input(self):
        assert _clean_evaluator_response("") == ""
        assert _clean_evaluator_response(None) is None
