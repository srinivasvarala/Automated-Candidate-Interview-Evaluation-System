from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.models import GraphState
from app.agents import interviewer_node, candidate_node, evaluator_node, summary_node


def _after_candidate(state: GraphState) -> str:
    """Route after candidate: if they requested end, go to summary; otherwise evaluate."""
    if state.get("end_requested"):
        return "summary"
    return "evaluator"


def build_interview_graph():
    """Construct and compile the interview state graph with in-memory checkpointer."""
    graph = StateGraph(GraphState)

    graph.add_node("interviewer", interviewer_node)
    graph.add_node("candidate", candidate_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("summary", summary_node)

    graph.add_edge(START, "interviewer")
    graph.add_edge("interviewer", "candidate")
    graph.add_conditional_edges("candidate", _after_candidate, {
        "evaluator": "evaluator",
        "summary": "summary",
    })
    # Evaluator always loops back to interviewer
    graph.add_edge("evaluator", "interviewer")
    graph.add_edge("summary", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


interview_graph = build_interview_graph()
