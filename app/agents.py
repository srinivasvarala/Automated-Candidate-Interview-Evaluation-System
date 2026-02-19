import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.config import settings
from app.models import GraphState
from app import llm_service


def _build_messages(state: GraphState, system_prompt: str) -> list[dict]:
    """Convert graph state messages + a system prompt into LiteLLM format."""
    msgs = [{"role": "system", "content": system_prompt}]
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msgs.append({"role": "assistant", "content": m.content})
        elif isinstance(m, SystemMessage):
            pass  # skip internal system messages in conversation
    return msgs


async def interviewer_node(state: GraphState) -> dict:
    """Generate the next interview question."""
    job = state["job_position"]
    q_num = state["questions_asked"] + 1

    system_prompt = f"""You are a professional interviewer for a {job} position.
You are asking question number {q_num}.

Guidelines:
- Ask ONE clear, specific question at a time
- Vary question types: technical, problem-solving, behavioral, culture fit
- Do not repeat topics already covered in the conversation
- Keep the question under 50 words
- Do NOT include any preamble like "Here is my question". Just ask the question directly."""

    response = await llm_service.complete(_build_messages(state, system_prompt))

    return {
        "messages": [AIMessage(content=response, name="interviewer")],
        "current_phase": "candidate",
    }


async def candidate_node(state: GraphState) -> dict:
    """Pause execution and wait for human input via interrupt."""
    answer = interrupt("waiting_for_input")

    # Check if user wants to end the interview
    if answer == "__END_INTERVIEW__":
        return {
            "messages": [HumanMessage(content="The candidate has ended the interview.", name="candidate")],
            "current_phase": "summary",
            "end_requested": True,
        }

    return {
        "messages": [HumanMessage(content=answer, name="candidate")],
        "current_phase": "evaluator",
        "end_requested": False,
    }


async def evaluator_node(state: GraphState) -> dict:
    """Evaluate the candidate's most recent answer."""
    job = state["job_position"]
    max_words = settings.max_evaluator_words

    system_prompt = f"""You are a career coach evaluating interview answers for a {job} position.

Guidelines:
- Give brief, constructive feedback on the candidate's MOST RECENT answer only (max {max_words} words)
- Mention one strength and one area for improvement
- Be encouraging but honest
- NEVER ask questions, follow-up questions, or prompts like "Can you elaborate?" or "What do you think?"
- NEVER pose hypothetical scenarios
- Your response must ONLY contain feedback — nothing else"""

    response = await llm_service.complete(_build_messages(state, system_prompt))

    return {
        "messages": [AIMessage(content=response, name="evaluator")],
        "questions_asked": state["questions_asked"] + 1,
        "current_phase": "interviewer",
    }


async def summary_node(state: GraphState) -> dict:
    """Generate a structured scorecard summarizing the interview."""
    job = state["job_position"]

    system_prompt = f"""You are an expert interview evaluator. Analyze the complete interview for a {job} position and produce a JSON scorecard.

Return ONLY valid JSON with this exact structure:
{{
  "scores": {{
    "technical": <1-10>,
    "problem_solving": <1-10>,
    "communication": <1-10>,
    "culture_fit": <1-10>
  }},
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "improvements": ["improvement 1", "improvement 2", "improvement 3"],
  "overall_summary": "2-3 sentence summary of the candidate's performance"
}}"""

    response = await llm_service.complete_json(_build_messages(state, system_prompt))

    try:
        summary_data = json.loads(response)
    except json.JSONDecodeError:
        summary_data = {
            "scores": {"technical": 5, "problem_solving": 5, "communication": 5, "culture_fit": 5},
            "strengths": ["Unable to parse detailed feedback"],
            "improvements": ["Unable to parse detailed feedback"],
            "overall_summary": response[:300],
        }

    return {
        "messages": [AIMessage(content=response, name="summary")],
        "summary_data": summary_data,
        "current_phase": "done",
    }
