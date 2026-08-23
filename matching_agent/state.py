from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class RetrievalSource(TypedDict):
    query: str
    query_kind: str
    rank: int
    score: float | None


class CandidateHit(TypedDict):
    username: str
    message: str
    sources: list[RetrievalSource]


class FailureLog(TypedDict):
    detail: str
    failed_query: str
    failed_matched_message: str
    failed_matched_candidate: str
    certainty: int


class TraceEvent(TypedDict):
    node: str
    detail: str


def merge_dicts(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """Merge parallel selector outputs using the selector name as the key."""
    return {**left, **right}


def append_unique_failures(
    left: list[FailureLog], right: list[FailureLog]
) -> list[FailureLog]:
    """Append failures without duplicating the same query/candidate pair."""
    merged = list(left)
    seen = {
        (item["failed_query"], item["failed_matched_candidate"])
        for item in merged
    }
    for item in right:
        key = (item["failed_query"], item["failed_matched_candidate"])
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


class AgentState(TypedDict, total=False):
    """Shared state for one matching run.

    Nodes return partial updates. Reducers are reserved for channels written by
    parallel selector nodes or accumulated across evaluator retry cycles.
    """

    input_message: str
    username: str
    hypernym: str
    matching_target: str
    analyzed_message: str | None
    search_info: str | None
    reformed_queries: list[str]
    reformed_query: str | None
    candidate_hits: list[CandidateHit]
    candidates: list[str]
    selector_results: Annotated[dict[str, str], merge_dicts]
    matched_message: str | None
    matched_username: str | None
    certainty: int | None
    fail_or_not: str
    evaluation_count: int
    retry_count: int
    next_agent: str
    last_agent: str
    messages: str
    history: Annotated[list[BaseMessage], add_messages]
    failure_log: Annotated[list[FailureLog], append_unique_failures]
    trace: Annotated[list[TraceEvent], list.__add__]
    metadata: dict[str, Any]


def get_initial_state(input_message: str, username: str = "Guest") -> AgentState:
    return {
        "input_message": input_message,
        "username": username,
        "hypernym": "",
        "matching_target": "",
        "analyzed_message": None,
        "search_info": None,
        "reformed_queries": [],
        "reformed_query": None,
        "candidate_hits": [],
        "candidates": [],
        "selector_results": {},
        "matched_message": None,
        "matched_username": None,
        "certainty": None,
        "fail_or_not": "",
        "evaluation_count": 0,
        "retry_count": 0,
        "next_agent": "Orchestrator",
        "last_agent": "User",
        "messages": input_message,
        "history": [HumanMessage(content=input_message)],
        "failure_log": [],
        "trace": [{"node": "User", "detail": input_message}],
        "metadata": {},
    }
