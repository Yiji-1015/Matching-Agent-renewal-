from __future__ import annotations

import unittest

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END
from langgraph.types import Command
from pydantic import ValidationError

from matching_agent.graph import build_matching_graph
from matching_agent.nodes import MatchingAgentNodes
from matching_agent.retriever import collect_candidate_hits
from matching_agent.schemas import (
    EvaluatorResponse,
    OrchestratorResponse,
    QueryReformerResponse,
    RoleSelectorResponse,
    SelectorResponse,
    SubSelectorResponse,
)
from matching_agent.state import get_initial_state


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def invoke(self, query: str):
        self.queries.append(query)
        shared = Document(page_content="같이 합주할 밴드를 찾습니다.", metadata={"User": "B"})
        own = Document(page_content="내 글", metadata={"User": "A"})
        unique = Document(page_content=f"{query} 후보", metadata={"User": query})
        return [shared, own, unique]


class FakeStructuredLLM:
    """Return schema-valid outputs while still exercising prompt formatting."""

    def with_structured_output(self, schema):
        responses = {
            OrchestratorResponse: lambda: OrchestratorResponse(
                hypernym="밴드 활동",
                matching_target="기타리스트",
                reasoning="입력이 명확하다",
                next_agent="Query_Reformer",
                next_action="쿼리를 재구성한다",
            ),
            QueryReformerResponse: lambda: QueryReformerResponse(
                queries=["밴드를 찾는 기타리스트", "합주팀을 찾는 연주자"],
                reasoning="역할을 반전했다",
                next_agent="Selector",
                next_action="후보를 검색한다",
            ),
            RoleSelectorResponse: lambda: RoleSelectorResponse(
                user_role="모집자",
                candidates=[
                    {
                        "candidate": "B",
                        "coherence_with_user": "지원자 역할",
                        "score": 0.9,
                    }
                ],
            ),
            SelectorResponse: lambda: SelectorResponse(
                matched_message="같이 합주할 밴드를 찾습니다.",
                matched_username="B",
                reasoning="모집자와 지원자로 상보적이다",
                next_agent="Evaluator",
                next_action="최종 평가한다",
                certainty=90,
            ),
            EvaluatorResponse: lambda: EvaluatorResponse(
                matched_username="B",
                matched_message="같이 합주할 밴드를 찾습니다.",
                fail_or_not="success",
                reasoning="상보성이 확인되었다",
                next_agent="__end__",
                next_action="종료한다",
                success_or_fail="success",
                detail="",
                failed_query="",
                matched_candidate="B",
                certainty=90,
            ),
            SubSelectorResponse: lambda: SubSelectorResponse(opinion="적합"),
        }
        return RunnableLambda(lambda _: responses[schema]())


class RetryingStructuredLLM(FakeStructuredLLM):
    def __init__(self):
        self.evaluator_calls = 0

    def with_structured_output(self, schema):
        if schema is not EvaluatorResponse:
            return super().with_structured_output(schema)

        def evaluate(_):
            self.evaluator_calls += 1
            failed = self.evaluator_calls == 1
            return EvaluatorResponse(
                matched_username="B",
                matched_message="같이 합주할 밴드를 찾습니다.",
                fail_or_not="fail" if failed else "success",
                reasoning="쿼리를 넓혀야 한다" if failed else "상보성이 확인되었다",
                next_agent="Orchestrator" if failed else "__end__",
                next_action="재탐색한다" if failed else "종료한다",
                success_or_fail="fail" if failed else "success",
                detail="역할 표현을 일반화한다" if failed else "",
                failed_query="밴드를 찾는 기타리스트" if failed else "",
                matched_candidate="B",
                certainty=70 if failed else 90,
            )

        return RunnableLambda(evaluate)


class UnusedWebSearchAgent:
    def invoke(self, state):
        raise AssertionError("web search should not run for an explicit input")


class FakeNodes:
    def orchestrator(self, state):
        return {
            "next_agent": "Web_Search_Module",
            "trace": [{"node": "Orchestrator", "detail": "optional search"}],
        }

    def web_search(self, state):
        return {
            "search_info": "고유명사 설명",
            "next_agent": "Message_Analyzer",
            "trace": [{"node": "Web_Search_Module", "detail": "searched"}],
        }

    def message_analyzer(self, state):
        return {
            "analyzed_message": "밴드 구성원을 찾는 모집 글",
            "next_agent": "Query_Reformer",
            "trace": [{"node": "Message_Analyzer", "detail": "analyzed"}],
        }

    def query_reformer(self, state):
        cycle = state.get("retry_count", 0)
        return {
            "reformed_queries": [f"밴드를 찾는 기타리스트 {cycle}"],
            "reformed_query": f"query-{cycle}",
            "trace": [{"node": "Query_Reformer", "detail": f"cycle={cycle}"}],
        }

    def retrieve(self, state):
        return {
            "candidates": ["B: 같이 합주할 밴드를 찾습니다."],
            "trace": [{"node": "Retrieve", "detail": "one candidate"}],
        }

    def type_match(self, state):
        return {
            "selector_results": {"type": "사람 타입 일치"},
            "trace": [{"node": "TypeMatch", "detail": "ok"}],
        }

    def role_match(self, state):
        return {
            "selector_results": {"role": "모집자-지원자 상보"},
            "trace": [{"node": "RoleMatch", "detail": "ok"}],
        }

    def persona_match(self, state):
        return {
            "selector_results": {"persona": "관심사 일치"},
            "trace": [{"node": "PersonaMatch", "detail": "ok"}],
        }

    def selector(self, state):
        assert set(state["selector_results"]) == {"type", "role", "persona"}
        return {
            "matched_username": "B",
            "matched_message": "같이 합주할 밴드를 찾습니다.",
            "certainty": 90,
            "trace": [{"node": "Selector", "detail": "selected"}],
        }

    def evaluator(self, state):
        evaluation = state.get("evaluation_count", 0) + 1
        if evaluation == 1:
            return Command(
                update={
                    "evaluation_count": evaluation,
                    "retry_count": 1,
                    "failure_log": [
                        {
                            "detail": "broaden query",
                            "failed_query": state["reformed_query"],
                            "failed_matched_message": state["matched_message"],
                            "failed_matched_candidate": state["matched_username"],
                            "certainty": state["certainty"],
                        }
                    ],
                    "trace": [{"node": "Evaluator", "detail": "retry"}],
                },
                goto="Query_Reformer",
            )
        return Command(
            update={
                "evaluation_count": evaluation,
                "fail_or_not": "success",
                "trace": [{"node": "Evaluator", "detail": "success"}],
            },
            goto=END,
        )


class RetrieverTests(unittest.TestCase):
    def test_multi_query_deduplication_preserves_provenance(self):
        hits = collect_candidate_hits(
            [("original", "원문"), ("reformed_1", "역할 반전")],
            "A",
            FakeRetriever(),
            k=3,
        )

        shared = next(hit for hit in hits if hit["username"] == "B")
        self.assertEqual(len(shared["sources"]), 2)
        self.assertEqual(
            [source["query_kind"] for source in shared["sources"]],
            ["original", "reformed_1"],
        )
        self.assertFalse(any(hit["username"] == "A" for hit in hits))

    def test_query_reformer_schema_requires_exactly_two_queries(self):
        with self.assertRaises(ValidationError):
            QueryReformerResponse(
                queries=["하나뿐인 쿼리"],
                reasoning="invalid",
                next_agent="Selector",
                next_action="retrieve",
            )
        with self.assertRaises(ValidationError):
            QueryReformerResponse(
                queries=["같은 쿼리", " 같은 쿼리 "],
                reasoning="invalid",
                next_agent="Selector",
                next_action="retrieve",
            )


class WorkflowTests(unittest.TestCase):
    def test_parallel_selectors_and_bounded_retry(self):
        graph = build_matching_graph(nodes=FakeNodes())
        result = graph.invoke(
            get_initial_state("기타리스트를 구합니다", "A"),
            {"configurable": {"thread_id": "test-workflow"}},
        )

        self.assertEqual(result["evaluation_count"], 2)
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(result["fail_or_not"], "success")
        self.assertEqual(len(result["failure_log"]), 1)
        self.assertEqual(
            set(result["selector_results"]), {"type", "role", "persona"}
        )
        self.assertEqual(
            sum(event["node"] == "Retrieve" for event in result["trace"]), 2
        )

    def test_real_nodes_format_prompts_and_complete_graph(self):
        retriever = FakeRetriever()
        nodes = MatchingAgentNodes(
            retriever,
            llm=FakeStructuredLLM(),
            web_search_agent=UnusedWebSearchAgent(),
        )
        graph = build_matching_graph(nodes=nodes)
        result = graph.invoke(
            get_initial_state("밴드에서 기타리스트를 구합니다", "A"),
            {"configurable": {"thread_id": "test-real-nodes"}},
        )

        self.assertEqual(result["fail_or_not"], "success")
        self.assertEqual(result["matched_username"], "B")
        self.assertEqual(result["evaluation_count"], 1)
        self.assertEqual(len(result["candidate_hits"]), 4)
        self.assertEqual(len(retriever.queries), 3)

    def test_real_evaluator_retries_once_then_succeeds(self):
        retriever = FakeRetriever()
        llm = RetryingStructuredLLM()
        nodes = MatchingAgentNodes(
            retriever,
            llm=llm,
            web_search_agent=UnusedWebSearchAgent(),
        )
        result = build_matching_graph(nodes=nodes).invoke(
            get_initial_state("밴드에서 기타리스트를 구합니다", "A"),
            {"configurable": {"thread_id": "test-real-retry"}},
        )

        self.assertEqual(result["evaluation_count"], 2)
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(result["fail_or_not"], "success")
        self.assertEqual(len(result["failure_log"]), 1)
        self.assertEqual(len(retriever.queries), 6)

    def test_missing_selector_result_stops_at_retry_limit(self):
        nodes = MatchingAgentNodes(
            FakeRetriever(),
            llm=FakeStructuredLLM(),
            web_search_agent=UnusedWebSearchAgent(),
            max_evaluations=2,
        )
        state = get_initial_state("밴드에서 기타리스트를 구합니다", "A")

        first = nodes.evaluator(state)
        self.assertEqual(first.goto, "Query_Reformer")

        state["retry_count"] = 1
        second = nodes.evaluator(state)
        self.assertEqual(second.goto, END)
        self.assertEqual(second.update["fail_or_not"], "fail")


if __name__ == "__main__":
    unittest.main()
