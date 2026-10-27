import json
import logging
import random

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.config import settings
from app.difficulty import Difficulty
from app.llm_output import AnswerScore, ScorecardOutput
from app.llm_service import LLMOutputError
from app.models import GraphState
from app import llm_service

logger = logging.getLogger(__name__)

# Interview type → question focus guidance with examples
_TYPE_GUIDANCE = {
    "behavioral": """Focus on behavioral questions that reveal real past experience. Demand specifics — names, numbers, timelines, outcomes.

Example good questions:
- "Walk me through a project where you had to push back on a stakeholder's technical decision. What was the outcome?"
- "Tell me about a deadline you missed. What went wrong and what did you change afterward?"
- "Describe a time you inherited a codebase with significant tech debt. How did you prioritize what to fix?"

Avoid cliché openers like "Tell me about a time when..." — instead, set up a specific scenario or constraint and ask how they handled something similar.""",

    "technical": """Focus on technical depth: problem-solving, architecture, debugging, and domain knowledge. Frame questions as concrete problems, not abstract theory.

Example good questions:
- "You deploy a change and latency spikes 3x on one endpoint but not others. Walk me through how you'd diagnose it."
- "How would you design the data model for a permission system that supports role inheritance and temporary access grants?"
- "What are the trade-offs between optimistic and pessimistic locking, and when would you choose each?"

Avoid trivia ("What does SOLID stand for?") and questions with a single correct answer. Prefer open-ended problems with trade-offs.""",

    "system_design": """Focus on system design and architecture. Present a concrete problem with scale constraints and ask the candidate to design a solution. Probe trade-offs.

Example good questions:
- "Design a notification system that delivers 50M push notifications per day with at-most-once delivery. Start with the high-level architecture."
- "How would you build a real-time collaborative document editor? What consistency model would you choose and why?"
- "You need to migrate a monolith to microservices without downtime. Walk me through your phased approach."

Always include a concrete scale number or constraint. Ask follow-ups about failure modes, bottlenecks, and what they'd monitor.""",

    "mixed": """Rotate question types in this pattern:
1. Behavioral (warm-up, motivation, teamwork)
2. Technical (domain knowledge, problem-solving)
3. Behavioral (leadership, conflict, failure)
4. Technical or System Design (architecture, trade-offs)
5+ Alternate between technical depth and scenario-based questions

This gives a well-rounded assessment. Don't cluster similar question types together.""",
}

_NEXT_QUESTION_PROMPT = "Ask the next interview question."


def _build_messages(
    state: GraphState, system_prompt: str, closing_prompt: str = "Please continue."
) -> list[dict]:
    """Convert graph state messages + a system prompt into LiteLLM format.

    Some models reject a conversation that ends on an assistant message
    (no prefill), so a closing user turn is added when that would happen.
    """
    msgs = [{"role": "system", "content": system_prompt}]
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msgs.append({"role": "assistant", "content": m.content})
        elif isinstance(m, SystemMessage):
            pass  # skip internal system messages in conversation
    if msgs[-1]["role"] == "assistant":
        msgs.append({"role": "user", "content": closing_prompt})
    return msgs


async def interviewer_node(state: GraphState) -> dict:
    """Generate the next interview question."""
    job = state["job_position"]
    jd = state.get("job_description", "")
    interview_type = state.get("interview_type", "mixed")
    q_num = state["questions_asked"] + 1
    total = settings.num_questions

    type_guidance = _TYPE_GUIDANCE.get(interview_type, _TYPE_GUIDANCE["mixed"])

    # Pick a random question angle to ensure variety across sessions
    _QUESTION_ANGLES = {
        "behavioral": [
            "teamwork and collaboration", "handling failure or setbacks", "leadership and initiative",
            "conflict resolution", "prioritization under pressure", "mentoring others",
            "adapting to change", "decision-making with incomplete information",
            "cross-functional communication", "dealing with ambiguity",
        ],
        "technical": [
            "debugging and root cause analysis", "system performance and optimization",
            "data modeling and storage decisions", "API design and contracts",
            "concurrency and race conditions", "testing strategy and trade-offs",
            "security considerations", "monitoring and observability",
            "code architecture decisions", "migration and backward compatibility",
        ],
        "system_design": [
            "scalability and load handling", "data consistency trade-offs",
            "failure modes and resilience", "caching strategy",
            "event-driven architecture", "service decomposition",
            "real-time vs batch processing", "cost optimization at scale",
            "observability and debugging distributed systems", "data pipeline design",
        ],
        "mixed": [
            "technical depth on a past project", "handling a difficult stakeholder",
            "debugging a production incident", "system design trade-offs",
            "teamwork under deadline pressure", "architecture decision-making",
            "learning a new technology quickly", "improving an existing system",
            "mentoring or knowledge sharing", "cross-team collaboration",
        ],
    }
    angles = _QUESTION_ANGLES.get(interview_type, _QUESTION_ANGLES["mixed"])
    random_angle = random.choice(angles)

    difficulty = Difficulty(state.get("scores") or [])

    system_prompt = f"""You are a senior engineer on the hiring committee, interviewing a candidate for a {job} position. You have 10+ years of industry experience and you've conducted hundreds of interviews. You're friendly but rigorous — you want to see how candidates actually think, not just whether they can recite textbook answers.

This is question {q_num} of the interview (target: ~{total} questions, but the candidate decides when to stop).

## Interview Focus
{type_guidance}

{difficulty.brief()}

## Opening Angle (for this question)
Start with a question exploring: **{random_angle}**. This is a suggestion to ensure variety — adapt it to the candidate's background and the conversation so far.

## Rules (non-negotiable)
- Ask exactly ONE question. Never combine two questions with "and" or "also".
- Keep the question concise, but include enough context or constraints to make it specific. A scenario setup of 1-2 sentences before the question is fine.
- Just ask the question directly. No preamble ("Great question to consider!", "Let's move on to..."), no commentary, no encouragement.
- Never repeat a topic already covered in this conversation. Check the conversation history.
- If the candidate gave an interesting or incomplete answer previously, you may ask a follow-up that digs deeper into that specific topic — great interviewers do this.
- **If the candidate asked for clarification, said they didn't understand, or asked you to elaborate on the previous question — DO NOT move to a new topic. Instead, rephrase or simplify the SAME question so they can attempt an answer. This is what a good interviewer does.**
- Avoid cliché formulations like "Tell me about a time when..." — rephrase to be more specific and engaging.
- Never ask trivia or questions with a single correct answer. Prefer questions with trade-offs, ambiguity, or multiple valid approaches."""

    if jd:
        system_prompt += f"""

## Job Description
Tailor your questions to probe the specific skills, tools, and responsibilities mentioned below. Prioritize areas that are hard to assess from a resume alone:

{jd[:3000]}"""

    resume = state.get("resume_text", "")
    if resume:
        system_prompt += f"""

## Candidate's Resume
The candidate submitted the resume below. Use it to:
- Ask about specific projects, roles, or claims they've made — verify depth vs. surface-level familiarity
- Probe gaps: skills required by the job description but absent from the resume
- If they claim leadership/architecture experience, ask for concrete details (team size, decisions made, outcomes)
- Do NOT summarize or reference the resume directly ("I see on your resume..."). Instead, ask naturally — e.g., "You worked on a distributed data pipeline — what was the hardest scaling challenge you hit?"

{resume[:4000]}"""

    logger.info("Interviewer generating question %d/%d for %s (%s)", q_num, total, job, interview_type)
    response = await llm_service.complete(_build_messages(state, system_prompt, _NEXT_QUESTION_PROMPT))

    # If LLM returned empty, retry once with higher temperature
    if not response or not response.strip():
        logger.warning("Interviewer got empty response, retrying with higher temperature")
        response = await llm_service.complete(
            _build_messages(state, system_prompt, _NEXT_QUESTION_PROMPT), temperature=0.9
        )

    if not response or not response.strip():
        logger.error("Interviewer failed to generate question after retry")
        response = f"What's a technical challenge you've faced in your work as a {job} that required you to make a difficult trade-off?"

    return {
        "messages": [AIMessage(content=response, name="interviewer")],
        "current_phase": "candidate",
    }


async def candidate_node(state: GraphState) -> dict:
    """Pause execution and wait for human input via interrupt."""
    answer = interrupt("waiting_for_input")

    if answer == "__END_INTERVIEW__":
        logger.info("Candidate ended interview early")
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


def _clean_evaluator_response(text: str) -> str:
    """Strip any trailing questions or extra content the evaluator shouldn't produce.

    The evaluator sometimes appends interview questions after its feedback.
    We keep only the feedback + stronger answer section and cut everything else.
    """
    if not text:
        return text

    import re

    # Find the "Stronger answer" section — everything after its paragraph is suspect
    stronger_match = re.search(r'💡\s*\*?\*?Stronger answer:?\*?\*?', text)
    if stronger_match:
        # Keep from start through the stronger answer paragraph
        after_marker = text[stronger_match.start():]
        # The stronger answer is typically 2-3 sentences ending with a quote or period.
        # Find the end: look for a double newline, horizontal rule, or a line that
        # starts a new question (contains "?" at the end of a line)
        lines = after_marker.split('\n')
        kept_lines = []
        found_content = False
        for line in lines:
            stripped = line.strip()
            # Stop at horizontal rules (--- or ___) that come after content
            if found_content and re.match(r'^[-_]{3,}\s*$', stripped):
                break
            # Stop at empty line followed by what looks like a new question
            if found_content and not stripped:
                kept_lines.append(line)
                continue
            # Stop if we hit a line that looks like a new interview question
            # (long sentence ending with ?)
            if found_content and stripped.endswith('?') and len(stripped) > 40:
                # Check if this is part of the stronger answer or a new question
                if '"' not in stripped and "'" not in stripped:
                    break
            if stripped:
                found_content = True
            kept_lines.append(line)

        # Trim trailing blank lines
        while kept_lines and not kept_lines[-1].strip():
            kept_lines.pop()

        cleaned = text[:stronger_match.start()] + '\n'.join(kept_lines)
    else:
        cleaned = text

    # Final safety: remove any trailing paragraph that ends with "?" and is > 40 chars
    # (likely an appended interview question)
    paragraphs = cleaned.rstrip().rsplit('\n\n', 1)
    if len(paragraphs) == 2:
        last_para = paragraphs[1].strip()
        if last_para.endswith('?') and len(last_para) > 40 and '"' not in last_para:
            cleaned = paragraphs[0]

    return cleaned.rstrip()


async def evaluator_node(state: GraphState) -> dict:
    """Evaluate the candidate's most recent answer and produce an adaptive score."""
    job = state["job_position"]
    max_words = settings.max_evaluator_words
    q_num = state["questions_asked"] + 1
    difficulty = Difficulty(state.get("scores") or [])

    # Step 1: Get the text feedback (shown to user)
    feedback_prompt = f"""You are a career coach evaluating interview answers for a {job} position.

Your role is ONLY to evaluate the candidate's answer. You are NOT the interviewer.

## What you MUST do:
- Give brief, constructive feedback on the candidate's MOST RECENT answer only (max {max_words} words)
- Mention one strength and one area for improvement
- Be encouraging but honest
- End with a "💡 **Stronger answer:**" section showing a 2-3 sentence example of a better response

## What you must NEVER do (violations will break the system):
- NEVER ask ANY question — no follow-ups, no "Can you elaborate?", no "What do you think?", no "Tell me about..."
- NEVER pose a new interview question or scenario
- NEVER rephrase or re-explain the interviewer's question
- Your response must contain ZERO question marks except inside the "Stronger answer" example
- If the candidate asked for clarification instead of answering, note that and advise how to handle unclear questions in real interviews. Do NOT provide the answer.

## Response format (follow exactly):
[2-4 sentences of feedback — NO questions]

💡 **Stronger answer:** [2-3 sentence example of a better response]"""

    logger.info("Evaluator analyzing answer for question %d", q_num)
    raw_feedback = await llm_service.complete(_build_messages(state, feedback_prompt))
    feedback_response = _clean_evaluator_response(raw_feedback)

    # Step 2: Get a quick numeric score (not shown to user, feeds adaptive engine)
    score_prompt = f"""Rate the candidate's MOST RECENT answer on a scale of 1-10 for a {job} position.

Consider: accuracy, depth, specificity, structure, and relevance.
- 1-3: Weak (vague, incorrect, or off-topic)
- 4-6: Adequate (partially correct, lacks depth or examples)
- 7-8: Strong (well-structured, specific, demonstrates real understanding)
- 9-10: Exceptional (insightful, goes beyond the question, expert-level)

Return ONLY a JSON object: {{"score": <1-10>}}"""

    try:
        score = (await llm_service.complete_model(_build_messages(state, score_prompt), AnswerScore)).score
    except LLMOutputError:
        logger.warning("Evaluator score unavailable; leaving this Turn unscored")
        score = None

    updated = difficulty.record(score)
    logger.info("Q%d score=%s, tier=%s, next_tier=%s",
                q_num, score, updated.curve()[-1]["tier"], updated.tier.name)

    return {
        "messages": [AIMessage(content=feedback_response or "Good attempt. Try to be more specific with concrete examples.", name="evaluator")],
        "questions_asked": q_num,
        "current_phase": "interviewer",
        "scores": list(updated.scores),
    }


async def summary_node(state: GraphState) -> dict:
    """Generate a structured scorecard summarizing the interview."""
    job = state["job_position"]
    resume = state.get("resume_text", "")
    resume_analysis_block = ""
    resume_json_field = ""
    if resume:
        resume_analysis_block = f"""

## Resume Context
The candidate submitted the following resume. Compare their interview performance against their resume claims:

{resume[:4000]}

When generating the scorecard, include a "resume_analysis" field that assesses:
- **verified_skills**: Skills/experience from the resume that the candidate demonstrated genuine depth in during the interview (2-4 items)
- **gaps**: Skills claimed on the resume where the candidate's answers revealed shallow or surface-level knowledge (1-3 items)
- **surprises**: Strengths or knowledge the candidate showed that WEREN'T on their resume (0-2 items)"""

        resume_json_field = """,
  "resume_analysis": {
    "verified_skills": ["skill the candidate proved they truly know"],
    "gaps": ["resume claim that interview answers didn't support"],
    "surprises": ["unexpected strength not on resume"]
  }"""

    system_prompt = f"""You are an expert interview evaluator. Analyze the complete interview for a {job} position and produce a JSON scorecard.
{resume_analysis_block}

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
  "overall_summary": "2-3 sentence summary of the candidate's performance"{resume_json_field}
}}"""

    logger.info("Summary node generating scorecard")
    try:
        scorecard = await llm_service.complete_model(_build_messages(state, system_prompt), ScorecardOutput)
        summary_data = scorecard.model_dump(exclude_none=True)
        content = json.dumps(summary_data)
    except LLMOutputError:
        logger.warning("Scorecard unavailable; delivering the difficulty curve only")
        content = "We couldn't generate your detailed scorecard."
        summary_data = {"scores": {}, "strengths": [], "improvements": [], "overall_summary": content}

    # Attach difficulty curve data (not from LLM — computed from the scores)
    summary_data["difficulty_curve"] = Difficulty(state.get("scores") or []).curve()

    return {
        "messages": [AIMessage(content=content, name="summary")],
        "summary_data": summary_data,
        "current_phase": "done",
    }
