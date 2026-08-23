from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


class OrchestratorResponse(BaseModel):
    hypernym: Optional[str]
    matching_target: Optional[str]
    reasoning: str
    next_agent: Literal[
        "Web_Search_Module",
        "Message_Analyzer",
        "Query_Reformer",
        "Selector",
        "Evaluator",
        "__end__",
    ]
    next_action: str


class MessageAnalyzerResponse(BaseModel):
    hypernym: Optional[str]
    matching_target: Optional[str]
    reasoning: str
    next_agent: Literal["Orchestrator", "Web_Search_Module"]
    next_action: str


class QueryReformerResponse(BaseModel):
    queries: Annotated[
        list[Annotated[str, Field(min_length=1)]],
        Field(min_length=2, max_length=2),
    ]
    reasoning: str
    next_agent: Literal["Selector", "Web_Search_Module"]
    next_action: str

    @field_validator("queries")
    @classmethod
    def normalize_queries(cls, queries: list[str]) -> list[str]:
        normalized = [query.strip() for query in queries]
        if any(not query for query in normalized):
            raise ValueError("reformed queries must not be blank")
        if len({query.casefold() for query in normalized}) != 2:
            raise ValueError("reformed queries must be distinct")
        return normalized


class SelectorScore(TypedDict):
    candidate: str
    coherence_with_user: str
    score: float


class SubSelectorResponse(BaseModel):
    opinion: str


class RoleSelectorResponse(BaseModel):
    user_role: str
    candidates: list[SelectorScore]


class SelectorResponse(BaseModel):
    matched_message: str
    matched_username: str
    reasoning: str
    next_agent: Literal["Evaluator", "Orchestrator"]
    next_action: str
    certainty: int


class EvaluatorResponse(BaseModel):
    matched_username: str
    matched_message: str
    fail_or_not: str
    reasoning: str
    next_agent: Literal["Orchestrator", "__end__"]
    next_action: str
    success_or_fail: str
    detail: str
    failed_query: str
    matched_candidate: str
    certainty: int
