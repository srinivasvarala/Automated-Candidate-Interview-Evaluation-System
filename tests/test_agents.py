from unittest.mock import patch, AsyncMock

import pytest
from langchain_core.messages import AIMessage

from app.agents import (
    interviewer_node,
    evaluator_node,
    summary_node,
    _build_messages,
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


    def test_never_ends_on_an_assistant_message(self, state_with_conversation):
        state_with_conversation["messages"].append(AIMessage(content="Nice work.", name="evaluator"))
        msgs = _build_messages(state_with_conversation, "System prompt", "Ask the next question.")
        assert msgs[-2] == {"role": "assistant", "content": "Nice work."}
        assert msgs[-1] == {"role": "user", "content": "Ask the next question."}

    def test_leaves_a_trailing_user_message_alone(self, state_with_conversation):
        msgs = _build_messages(state_with_conversation, "System prompt")
        assert msgs[-1]["role"] == "user"
        assert "Please continue" not in msgs[-1]["content"]


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

    @pytest.mark.asyncio
    async def test_prep_focus_is_added_to_the_prompt_when_set(self, sample_state, mock_llm_response):
        sample_state["prep_focus"] = "Weakest areas so far (ask about these first):\n- Observability: tracing a RAG bot"
        with patch("app.agents.llm_service.complete", mock_llm_response) as mock:
            await interviewer_node(sample_state)

        system_msg = mock.call_args[0][0][0]["content"]
        assert "## Prep Focus" in system_msg
        assert "tracing a RAG bot" in system_msg

    @pytest.mark.asyncio
    async def test_no_prep_focus_section_without_a_plan(self, sample_state, mock_llm_response):
        with patch("app.agents.llm_service.complete", mock_llm_response) as mock:
            await interviewer_node(sample_state)

        assert "Prep Focus" not in mock.call_args[0][0][0]["content"]


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
    async def test_records_the_score(self, state_with_conversation, mock_llm_response, mock_llm_score_response):
        with patch("app.agents.llm_service.complete", mock_llm_response), \
             patch("app.agents.llm_service.complete_json", mock_llm_score_response):
            result = await evaluator_node(state_with_conversation)

        assert result["scores"] == [7]

    @pytest.mark.asyncio
    async def test_unparseable_score_leaves_the_turn_unscored(self, state_with_conversation, mock_llm_response):
        bad_score = AsyncMock(return_value="not json")
        with patch("app.agents.llm_service.complete", mock_llm_response), \
             patch("app.agents.llm_service.complete_json", bad_score):
            result = await evaluator_node(state_with_conversation)

        assert bad_score.call_count == 2  # one retry
        assert result["scores"] == [None]
        assert result["messages"][0].name == "evaluator"  # feedback is still delivered

    @pytest.mark.asyncio
    async def test_retry_can_recover_the_score(self, state_with_conversation, mock_llm_response):
        flaky = AsyncMock(side_effect=["oops", '{"score": 8}'])
        with patch("app.agents.llm_service.complete", mock_llm_response), \
             patch("app.agents.llm_service.complete_json", flaky):
            result = await evaluator_node(state_with_conversation)

        assert result["scores"] == [8]


class TestSummaryNode:
    @pytest.mark.asyncio
    async def test_parses_valid_json(self, state_with_conversation, mock_llm_json_response):
        with patch("app.agents.llm_service.complete_json", mock_llm_json_response):
            result = await summary_node(state_with_conversation)

        assert result["summary_data"]["scores"]["technical"] == 7
        assert len(result["summary_data"]["strengths"]) == 3
        assert result["current_phase"] == "done"

    @pytest.mark.asyncio
    async def test_unparseable_scorecard_still_delivers_the_curve(self, state_with_conversation):
        state_with_conversation["scores"] = [8, 3]
        bad_mock = AsyncMock(return_value="This is not JSON at all")
        with patch("app.agents.llm_service.complete_json", bad_mock):
            result = await summary_node(state_with_conversation)

        data = result["summary_data"]
        assert data["scores"] == {}
        assert "couldn't generate" in data["overall_summary"]
        assert [q["score"] for q in data["difficulty_curve"]] == [8, 3]
        assert result["messages"][0].content == data["overall_summary"]
        assert result["current_phase"] == "done"

    @pytest.mark.asyncio
    async def test_scorecard_keeps_valid_categories_only(self, state_with_conversation):
        reply = AsyncMock(return_value='{"scores": {"technical": 7, "communication": 42}, "overall_summary": "ok"}')
        with patch("app.agents.llm_service.complete_json", reply):
            result = await summary_node(state_with_conversation)

        assert result["summary_data"]["scores"] == {"technical": 7}


    @pytest.mark.asyncio
    async def test_returns_summary_ai_message(self, state_with_conversation, mock_llm_json_response):
        with patch("app.agents.llm_service.complete_json", mock_llm_json_response):
            result = await summary_node(state_with_conversation)

        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].name == "summary"


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
