from __future__ import annotations

from typing import Literal

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.types import Command

from .config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_RETRIEVAL_K,
)
from .prompts import load_prompt
from .retriever import collect_candidate_hits, format_candidates
from .schemas import (
    EvaluatorResponse,
    MessageAnalyzerResponse,
    OrchestratorResponse,
    QueryReformerResponse,
    RoleSelectorResponse,
    SelectorResponse,
    SubSelectorResponse,
)
from .state import AgentState, FailureLog


@tool
def web_search_tool(query: str):
    """Search the web for a proper noun or expression needed for matching."""
    web_search = TavilySearchResults(max_results=5)
    search_results = web_search.invoke(query)
    return [item.get("content") for item in search_results]


class MatchingAgentNodes:
    """LangGraph node implementations with injectable external dependencies."""

    def __init__(
        self,
        retriever,
        *,
        llm_model: str = DEFAULT_LLM_MODEL,
        llm=None,
        web_search_agent=None,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
    ):
        self.retriever = retriever
        self.gpt = llm or ChatOpenAI(model=llm_model, temperature=0)
        self.retrieval_k = retrieval_k
        self.max_evaluations = max_evaluations

        self.orchestrator_prompt = load_prompt("Orchestrator.txt")
        self.message_analyzer_prompt = load_prompt("Message_Analyzer.txt")
        self.query_reformer_prompt = load_prompt("Query_Reformer.txt")
        self.persona_prompt = load_prompt("Selector_PersonaMatch.txt")
        self.role_prompt = load_prompt("Selector_RoleMatch.txt")
        self.type_prompt = load_prompt("Selector_TypeMatch.txt")
        self.selector_prompt = load_prompt("Selector.txt")
        self.evaluator_prompt = load_prompt("Evaluator.txt")
        self.web_search_prompt = load_prompt("Web_Search_Module.txt")

        self.web_search_agent = web_search_agent or create_agent(
            model=self.gpt,
            tools=[web_search_tool],
            system_prompt=self.web_search_prompt,
            name="matching_web_search",
        )

    def _structured_invoke(
        self,
        prompt_text: str,
        response_schema,
        state: AgentState,
        **partial_values,
    ):
        prompt = ChatPromptTemplate.from_messages([("system", prompt_text)])
        if partial_values:
            prompt = prompt.partial(**partial_values)
        return (prompt | self.gpt.with_structured_output(response_schema)).invoke(state)

    @staticmethod
    def _event(node: str, detail: str):
        return [{"node": node, "detail": detail}]

    def orchestrator(self, state: AgentState) -> dict:
        """Interpret the input and choose only optional enrichment work.

        The graph, not the LLM, guarantees reformulation, retrieval, selection,
        and evaluation. This keeps the research workflow deterministic.
        """
        response = self._structured_invoke(
            self.orchestrator_prompt,
            OrchestratorResponse,
            state,
            recent_history=state.get("history", [])[-5:],
            analyzed_message=state.get("analyzed_message"),
            search_info=state.get("search_info"),
        )

        requested = response.next_agent
        if requested == "Web_Search_Module" and not state.get("search_info"):
            route = "Web_Search_Module"
        elif requested == "Message_Analyzer" and not state.get("analyzed_message"):
            route = "Message_Analyzer"
        else:
            route = "Query_Reformer"

        detail = f"{response.reasoning}\n{response.next_action}"
        return {
            "hypernym": response.hypernym or state.get("hypernym", ""),
            "matching_target": response.matching_target
            or state.get("matching_target", ""),
            "next_agent": route,
            "last_agent": "Orchestrator",
            "messages": detail,
            "history": [AIMessage(content=detail, name="Orchestrator")],
            "trace": self._event("Orchestrator", f"route={route}: {detail}"),
        }

    def message_analyzer(self, state: AgentState) -> dict:
        response = self._structured_invoke(
            self.message_analyzer_prompt,
            MessageAnalyzerResponse,
            state,
            recent_history=state.get("history", [])[-5:],
        )
        route = (
            "Web_Search_Module"
            if response.next_agent == "Web_Search_Module"
            and not state.get("search_info")
            else "Query_Reformer"
        )
        detail = f"{response.reasoning}\n{response.next_action}"
        return {
            "hypernym": response.hypernym or state.get("hypernym", ""),
            "matching_target": response.matching_target
            or state.get("matching_target", ""),
            "analyzed_message": detail,
            "next_agent": route,
            "last_agent": "Message_Analyzer",
            "messages": detail,
            "history": [AIMessage(content=detail, name="Message_Analyzer")],
            "trace": self._event("Message_Analyzer", f"route={route}: {detail}"),
        }

    def web_search(self, state: AgentState) -> dict:
        response = self.web_search_agent.invoke(
            {"messages": [HumanMessage(content=state["input_message"])]}
        )
        detail = response["messages"][-1].content
        route = (
            "Message_Analyzer"
            if not state.get("analyzed_message")
            else "Query_Reformer"
        )
        return {
            "search_info": detail,
            "next_agent": route,
            "last_agent": "Web_Search_Module",
            "messages": detail,
            "history": [AIMessage(content=detail, name="Web_Search_Module")],
            "trace": self._event("Web_Search_Module", f"route={route}: {detail}"),
        }

    def query_reformer(self, state: AgentState) -> dict:
        response = self._structured_invoke(
            self.query_reformer_prompt,
            QueryReformerResponse,
            state,
        )
        queries = [query.strip() for query in response.queries]
        reformed_query = " / ".join(queries)
        detail = (
            f"Reformed Queries - {reformed_query}, "
            f"Reasoning - {response.reasoning}\n{response.next_action}"
        )
        route = (
            "Web_Search_Module"
            if response.next_agent == "Web_Search_Module"
            and not state.get("search_info")
            else "Retrieve"
        )
        return {
            "reformed_queries": queries,
            "reformed_query": reformed_query,
            "candidate_hits": [],
            "candidates": [],
            "matched_message": None,
            "matched_username": None,
            "certainty": None,
            "next_agent": route,
            "last_agent": "Query_Reformer",
            "messages": detail,
            "history": [AIMessage(content=detail, name="Query_Reformer")],
            "trace": self._event("Query_Reformer", f"route={route}: {detail}"),
        }

    def retrieve(self, state: AgentState) -> dict:
        queries = [("original", state["input_message"])]
        queries.extend(
            (f"reformed_{index}", query)
            for index, query in enumerate(state.get("reformed_queries", []), start=1)
        )
        hits = collect_candidate_hits(
            queries,
            state["username"],
            self.retriever,
            k=self.retrieval_k,
        )
        candidates = format_candidates(hits)
        return {
            "candidate_hits": hits,
            "candidates": candidates,
            "next_agent": "Selectors",
            "last_agent": "Retrieve",
            "messages": f"Retrieved {len(candidates)} unique candidates.",
            "trace": self._event(
                "Retrieve",
                f"queries={len(queries)}, unique_candidates={len(candidates)}",
            ),
        }

    def persona_match(self, state: AgentState) -> dict:
        response = self._structured_invoke(
            self.persona_prompt, SubSelectorResponse, state
        )
        return {
            "selector_results": {"persona": response.opinion},
            "trace": self._event("PersonaMatch", response.opinion),
        }

    def role_match(self, state: AgentState) -> dict:
        response = self._structured_invoke(
            self.role_prompt, RoleSelectorResponse, state
        )
        opinion = f"사용자 역할: {response.user_role}\n후보 평가: {response.candidates}"
        return {
            "selector_results": {"role": opinion},
            "trace": self._event("RoleMatch", opinion),
        }

    def type_match(self, state: AgentState) -> dict:
        response = self._structured_invoke(
            self.type_prompt, SubSelectorResponse, state
        )
        return {
            "selector_results": {"type": response.opinion},
            "trace": self._event("TypeMatch", response.opinion),
        }

    @staticmethod
    def _selector_history(state: AgentState) -> list[str]:
        results = state.get("selector_results", {})
        return [
            f"This is from TypeMatch\n{results.get('type', '')}",
            f"This is from RoleMatch\n{results.get('role', '')}",
            f"This is from PersonaMatch\n{results.get('persona', '')}",
        ]

    def selector(self, state: AgentState) -> dict:
        invocation_state = dict(state)
        invocation_state["selectors_history"] = self._selector_history(state)
        response = self._structured_invoke(
            self.selector_prompt,
            SelectorResponse,
            invocation_state,
        )
        detail = f"{response.reasoning} {response.next_action}"
        return {
            "matched_message": response.matched_message,
            "matched_username": response.matched_username,
            "certainty": response.certainty,
            "next_agent": "Evaluator",
            "last_agent": "Selector",
            "messages": detail,
            "history": [AIMessage(content=detail, name="Selector")],
            "trace": self._event("Selector", detail),
        }

    def evaluator(
        self, state: AgentState
    ) -> Command[Literal["Query_Reformer", "__end__"]]:
        if state.get("certainty") is None:
            retry_count = state.get("retry_count", 0) + 1
            exhausted = retry_count >= self.max_evaluations
            goto = END if exhausted else "Query_Reformer"
            return Command(
                update={
                    "retry_count": retry_count,
                    "fail_or_not": "fail" if exhausted else state.get("fail_or_not", ""),
                    "next_agent": goto,
                    "trace": self._event(
                        "Evaluator",
                        "No selector result; retry limit reached."
                        if exhausted
                        else "No selector result; retrying reformulation.",
                    ),
                },
                goto=goto,
            )

        evaluation_number = state.get("evaluation_count", 0) + 1
        invocation_state = dict(state)
        invocation_state.update(
            {
                "evaluation_count": evaluation_number,
                "selectors_history": self._selector_history(state),
            }
        )
        response = self._structured_invoke(
            self.evaluator_prompt,
            EvaluatorResponse,
            invocation_state,
            chat_history=state.get("history", [])[-10:],
            candidates=state.get("candidates", []),
        )

        detail = f"{response.reasoning}\n{response.next_action}"
        update = {
            "fail_or_not": response.fail_or_not,
            "evaluation_count": evaluation_number,
            "last_agent": "Evaluator",
            "messages": detail,
            "matched_username": response.matched_username,
            "matched_message": response.matched_message,
            "history": [AIMessage(content=detail, name="Evaluator")],
            "trace": self._event(
                "Evaluator", f"evaluation={evaluation_number}: {detail}"
            ),
        }

        failed = response.fail_or_not == "fail"
        if failed:
            failure: FailureLog = {
                "detail": f"{response.reasoning}\n{response.detail}",
                "failed_query": response.failed_query,
                "failed_matched_message": response.matched_message,
                "failed_matched_candidate": response.matched_candidate,
                "certainty": response.certainty,
            }
            update["failure_log"] = [failure]

        if failed and evaluation_number < self.max_evaluations:
            update["retry_count"] = state.get("retry_count", 0) + 1
            update["next_agent"] = "Query_Reformer"
            return Command(update=update, goto="Query_Reformer")

        update["next_agent"] = END
        return Command(update=update, goto=END)
