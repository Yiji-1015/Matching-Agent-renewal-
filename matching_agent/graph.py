from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import RetryPolicy

from .nodes import MatchingAgentNodes
from .retriever import load_retriever
from .state import AgentState


LLM_RETRY_POLICY = RetryPolicy(max_attempts=2, initial_interval=0.5)


def _route_optional_enrichment(state: AgentState) -> str:
    route = state.get("next_agent", "Query_Reformer")
    if route not in {"Web_Search_Module", "Message_Analyzer", "Query_Reformer"}:
        return "Query_Reformer"
    return route


def _route_after_web_search(state: AgentState) -> str:
    return (
        "Message_Analyzer"
        if state.get("next_agent") == "Message_Analyzer"
        else "Query_Reformer"
    )


def _route_after_analysis(state: AgentState) -> str:
    return (
        "Web_Search_Module"
        if state.get("next_agent") == "Web_Search_Module"
        else "Query_Reformer"
    )


def _route_after_reformulation(state: AgentState) -> str:
    return (
        "Web_Search_Module"
        if state.get("next_agent") == "Web_Search_Module"
        else "Retrieve"
    )


def build_matching_graph(
    *,
    retriever=None,
    checkpointer=None,
    nodes: MatchingAgentNodes | None = None,
):
    """Build the matching workflow.

    The graph guarantees the core pipeline. The LLM router may request optional
    analysis or web enrichment, but cannot skip retrieval, parallel scoring,
    final selection, or evaluation.
    """
    if nodes is None:
        retriever = retriever or load_retriever()
        nodes = MatchingAgentNodes(retriever)

    builder = StateGraph(AgentState)
    builder.add_node(
        "Orchestrator", nodes.orchestrator, retry_policy=LLM_RETRY_POLICY
    )
    builder.add_node(
        "Web_Search_Module", nodes.web_search, retry_policy=LLM_RETRY_POLICY
    )
    builder.add_node(
        "Message_Analyzer", nodes.message_analyzer, retry_policy=LLM_RETRY_POLICY
    )
    builder.add_node(
        "Query_Reformer", nodes.query_reformer, retry_policy=LLM_RETRY_POLICY
    )
    builder.add_node("Retrieve", nodes.retrieve)

    # Fan out in one super-step. selector_results uses a dict reducer, so the
    # branches can safely write their named assessments concurrently.
    builder.add_node("TypeMatch", nodes.type_match, retry_policy=LLM_RETRY_POLICY)
    builder.add_node("RoleMatch", nodes.role_match, retry_policy=LLM_RETRY_POLICY)
    builder.add_node(
        "PersonaMatch", nodes.persona_match, retry_policy=LLM_RETRY_POLICY
    )
    builder.add_node("Selector", nodes.selector, retry_policy=LLM_RETRY_POLICY)

    # Evaluator returns Command(update=..., goto=...) and owns the bounded loop.
    builder.add_node("Evaluator", nodes.evaluator, retry_policy=LLM_RETRY_POLICY)

    builder.add_edge(START, "Orchestrator")
    builder.add_conditional_edges(
        "Orchestrator",
        _route_optional_enrichment,
        {
            "Web_Search_Module": "Web_Search_Module",
            "Message_Analyzer": "Message_Analyzer",
            "Query_Reformer": "Query_Reformer",
        },
    )
    builder.add_conditional_edges(
        "Web_Search_Module",
        _route_after_web_search,
        {
            "Message_Analyzer": "Message_Analyzer",
            "Query_Reformer": "Query_Reformer",
        },
    )
    builder.add_conditional_edges(
        "Message_Analyzer",
        _route_after_analysis,
        {
            "Web_Search_Module": "Web_Search_Module",
            "Query_Reformer": "Query_Reformer",
        },
    )
    builder.add_conditional_edges(
        "Query_Reformer",
        _route_after_reformulation,
        {
            "Web_Search_Module": "Web_Search_Module",
            "Retrieve": "Retrieve",
        },
    )
    builder.add_edge("Retrieve", "TypeMatch")
    builder.add_edge("Retrieve", "RoleMatch")
    builder.add_edge("Retrieve", "PersonaMatch")
    builder.add_edge(["TypeMatch", "RoleMatch", "PersonaMatch"], "Selector")
    builder.add_edge("Selector", "Evaluator")

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
