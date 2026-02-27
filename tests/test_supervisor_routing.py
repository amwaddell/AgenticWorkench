"""
Tests for the supervisor graph (Day 7).

Uses fake tools and a fake model — no LanceDB, no model server,
no heavy imports.  Validates:

- Heuristic router classifies obvious cases correctly
- LLM router fallback works
- Hybrid routing calls LLM only when heuristic is ambiguous
- Graph compiles with all/minimal tools
- invoke() runs the full pipeline and returns correct state
- Route dispatches to the correct subgraph
- Math, web, timeline routes produce expected results
- Error handling doesn't crash the graph
- Deterministic across invocations (heuristic_only)
"""

from __future__ import annotations

from typing import Any

import pytest

from workbench.core.types import ModelResponse

# Routing imports (from the new subpackage)
from workbench.graphs.routing import (
    DEFAULT_ROUTE,
    ROUTE_AMBIGUOUS,
    ROUTE_LOCAL_RAG,
    ROUTE_MATH,
    ROUTE_TIMELINE,
    ROUTE_WEB_RESEARCH,
    VALID_ROUTES,
    heuristic_route,
    llm_route,
)

# Graph builder
from workbench.graphs.supervisor_graph import build_supervisor_graph

# ------------------------------------------------------------------ #
#  Fakes                                                              #
# ------------------------------------------------------------------ #


class FakeSearchTool:
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "chunks": [
                {
                    "chunk_id": "chunk_001",
                    "score": 0.9,
                    "title": "Roman Republic",
                    "section": "Fall",
                    "snippet": "The Republic ended in 27 BC…",
                },
                {
                    "chunk_id": "chunk_002",
                    "score": 0.8,
                    "title": "Augustus",
                    "section": "Rise to power",
                    "snippet": "Octavian became Augustus…",
                },
            ]
        }


class FakeOpenChunkTool:
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        chunk_id = kwargs.get("chunk_id", "unknown")
        return {
            "found": True,
            "chunk": {
                "chunk_id": chunk_id,
                "title": "Test Article",
                "section": "Test Section",
                "text": f"Full text for {chunk_id}.",
            },
            "text_length": 30,
        }


class FakeTimelineTool:
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "timeline": [
                {
                    "date": "44 BC",
                    "event": "Caesar assassinated",
                    "supporting_chunk_ids": ["chunk_001"],
                },
                {
                    "date": "27 BC",
                    "event": "Republic ends",
                    "supporting_chunk_ids": ["chunk_001", "chunk_002"],
                },
            ],
            "raw_model_output": "...",
            "cited_chunk_ids": ["chunk_001", "chunk_002"],
        }


class FakeWebSearchTool:
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "results": [
                {
                    "title": "Latest News",
                    "url": "https://example.com/news",
                    "snippet": "Breaking news.",
                    "source": "duckduckgo",
                },
            ],
            "count": 1,
            "provider": "duckduckgo",
        }


class FakeCalculatorTool:
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        expression = kwargs.get("expression", "")
        try:
            if all(c in "0123456789+-*/. ()" for c in expression):
                result = eval(expression)  # noqa: S307 — test only
                return {"expression": expression, "result": result, "error": None}
        except Exception:
            pass
        return {"expression": expression, "result": None, "error": "Cannot evaluate"}


class FakeModel:
    def __init__(self, answer: str | None = None) -> None:
        self._answer = answer or (
            "The Roman Republic ended in 27 BC [chunk_001]. "
            "Augustus took power [chunk_002]."
        )

    def generate(self, messages: list, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            text=self._answer,
            tokens_in=100,
            tokens_out=50,
            latency_ms=42.0,
        )


class FakeRouterModel:
    """Returns a specific route label for LLM routing tests."""

    def __init__(self, route_label: str = "local_rag") -> None:
        self._route_label = route_label

    def generate(self, messages: list, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            text=self._route_label,
            tokens_in=20,
            tokens_out=5,
            latency_ms=10.0,
        )


class FailingModel:
    def generate(self, messages: list, **kwargs: Any) -> ModelResponse:
        raise RuntimeError("Model is down")


# ------------------------------------------------------------------ #
#  Heuristic router tests                                             #
# ------------------------------------------------------------------ #


class TestHeuristicRouter:
    def test_pure_math_expression(self) -> None:
        route, conf = heuristic_route("2 + 2 * 3")
        assert route == ROUTE_MATH
        assert conf >= 0.9

    def test_math_with_parens(self) -> None:
        route, _ = heuristic_route("(100 + 50) / 3")
        assert route == ROUTE_MATH

    def test_calculate_keyword(self) -> None:
        route, _ = heuristic_route("Calculate 17% of 340")
        assert route == ROUTE_MATH

    def test_what_is_with_number(self) -> None:
        route, _ = heuristic_route("What is 15 * 23?")
        assert route == ROUTE_MATH

    def test_solve_keyword(self) -> None:
        route, _ = heuristic_route("Solve 2^10")
        assert route == ROUTE_MATH

    def test_web_latest(self) -> None:
        route, _ = heuristic_route("What are the latest AI developments?")
        assert route == ROUTE_WEB_RESEARCH

    def test_web_today(self) -> None:
        route, _ = heuristic_route("What happened today in the stock market?")
        assert route == ROUTE_WEB_RESEARCH

    def test_web_current(self) -> None:
        route, _ = heuristic_route("Who is the current president of France?")
        assert route == ROUTE_WEB_RESEARCH

    def test_web_price(self) -> None:
        route, _ = heuristic_route("What is the price of Bitcoin?")
        assert route == ROUTE_WEB_RESEARCH

    def test_timeline_keyword(self) -> None:
        route, _ = heuristic_route("Give me a timeline of the French Revolution")
        assert route == ROUTE_TIMELINE

    def test_chronology_keyword(self) -> None:
        route, _ = heuristic_route("What is the chronology of World War II?")
        assert route == ROUTE_TIMELINE

    def test_when_did(self) -> None:
        route, _ = heuristic_route("When did the major events of the Cold War occur?")
        assert route == ROUTE_TIMELINE

    def test_history_of(self) -> None:
        route, _ = heuristic_route("What is the history of the Roman Empire?")
        assert route == ROUTE_TIMELINE

    def test_general_factual(self) -> None:
        route, _ = heuristic_route("What is photosynthesis?")
        assert route == ROUTE_LOCAL_RAG

    def test_who_question(self) -> None:
        route, _ = heuristic_route("Who was Alexander the Great?")
        assert route == ROUTE_LOCAL_RAG

    def test_ambiguous_statement(self) -> None:
        route, conf = heuristic_route("Tell me something interesting")
        assert conf <= 0.5

    def test_confidence_range(self) -> None:
        _, conf = heuristic_route("Calculate 5 + 3")
        assert 0.0 <= conf <= 1.0

    def test_case_insensitive(self) -> None:
        r1, _ = heuristic_route("CALCULATE 5 + 3")
        r2, _ = heuristic_route("calculate 5 + 3")
        assert r1 == r2 == ROUTE_MATH


# ------------------------------------------------------------------ #
#  LLM router tests                                                   #
# ------------------------------------------------------------------ #


class TestLLMRouter:
    def test_returns_valid_route(self) -> None:
        route, _ = llm_route("What happened today?", FakeRouterModel("web_research"))
        assert route == ROUTE_WEB_RESEARCH

    def test_returns_local_rag(self) -> None:
        route, _ = llm_route("What is photosynthesis?", FakeRouterModel("local_rag"))
        assert route == ROUTE_LOCAL_RAG

    def test_returns_math(self) -> None:
        route, _ = llm_route("Calculate 5*3", FakeRouterModel("math"))
        assert route == ROUTE_MATH

    def test_returns_timeline(self) -> None:
        route, _ = llm_route("Timeline of WWI", FakeRouterModel("timeline_research"))
        assert route == ROUTE_TIMELINE

    def test_invalid_label_falls_back(self) -> None:
        route, conf = llm_route("Something", FakeRouterModel("nonsense_route"))
        assert route == DEFAULT_ROUTE
        assert conf < 0.5

    def test_model_failure_falls_back(self) -> None:
        route, conf = llm_route("Something", FailingModel())
        assert route == DEFAULT_ROUTE
        assert conf < 0.5

    def test_confidence_moderate(self) -> None:
        _, conf = llm_route("What is gravity?", FakeRouterModel("local_rag"))
        assert 0.5 <= conf <= 1.0


# ------------------------------------------------------------------ #
#  Graph compilation tests                                            #
# ------------------------------------------------------------------ #


class TestGraphCompilation:
    def test_builds_with_all_tools(self) -> None:
        graph = build_supervisor_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            web_search_tool=FakeWebSearchTool(),
            timeline_tool=FakeTimelineTool(),
            calculator_tool=FakeCalculatorTool(),
        )
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_builds_with_minimal_tools(self) -> None:
        graph = build_supervisor_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
        )
        assert hasattr(graph, "invoke")

    def test_builds_without_calculator(self) -> None:
        graph = build_supervisor_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            web_search_tool=FakeWebSearchTool(),
        )
        assert hasattr(graph, "invoke")

    def test_builds_without_web_search(self) -> None:
        graph = build_supervisor_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            calculator_tool=FakeCalculatorTool(),
        )
        assert hasattr(graph, "invoke")

    def test_all_routing_modes(self) -> None:
        for mode in ("hybrid", "heuristic_only", "llm_only"):
            graph = build_supervisor_graph(
                search_tool=FakeSearchTool(),
                open_chunk_tool=FakeOpenChunkTool(),
                model=FakeModel(),
                routing_mode=mode,
            )
            assert hasattr(graph, "invoke")


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #


@pytest.fixture
def full_graph() -> Any:
    return build_supervisor_graph(
        search_tool=FakeSearchTool(),
        open_chunk_tool=FakeOpenChunkTool(),
        model=FakeModel(),
        web_search_tool=FakeWebSearchTool(),
        timeline_tool=FakeTimelineTool(),
        calculator_tool=FakeCalculatorTool(),
        routing_mode="heuristic_only",
    )


@pytest.fixture
def minimal_graph() -> Any:
    return build_supervisor_graph(
        search_tool=FakeSearchTool(),
        open_chunk_tool=FakeOpenChunkTool(),
        model=FakeModel(),
        routing_mode="heuristic_only",
    )


# ------------------------------------------------------------------ #
#  Graph invocation tests                                             #
# ------------------------------------------------------------------ #


class TestGraphInvocation:
    def test_returns_dict(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "What is photosynthesis?"})
        assert isinstance(result, dict)

    def test_has_answer_text(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "What is photosynthesis?"})
        assert len(result.get("answer_text", "")) > 0

    def test_has_route(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "What is photosynthesis?"})
        assert result["route"] in VALID_ROUTES

    def test_has_router_method(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "What is photosynthesis?"})
        assert result["router_method"] in ("heuristic", "llm")

    def test_has_router_confidence(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "What is photosynthesis?"})
        assert 0.0 <= result["router_confidence"] <= 1.0

    def test_has_subgraph_used(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "What is photosynthesis?"})
        assert "subgraph_used" in result

    def test_has_router_latency(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "What is photosynthesis?"})
        assert result["router_latency_ms"] >= 0


# ------------------------------------------------------------------ #
#  Route dispatch tests                                               #
# ------------------------------------------------------------------ #


class TestRouteDispatch:
    def test_math_routes_to_calculator(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "2 + 2 * 3"})
        assert result["route"] == ROUTE_MATH
        assert result["subgraph_used"] == "calculator"

    def test_web_routes_to_web_research(self, full_graph: Any) -> None:
        result = full_graph.invoke(
            {"question": "What are the latest news headlines today?"}
        )
        assert result["route"] == ROUTE_WEB_RESEARCH
        assert result["subgraph_used"] == "web_research"

    def test_timeline_routes_to_researcher(self, full_graph: Any) -> None:
        result = full_graph.invoke(
            {"question": "Give me a timeline of the Roman Republic"}
        )
        assert result["route"] == ROUTE_TIMELINE
        assert result["subgraph_used"] == "researcher_graph"

    def test_factual_routes_to_local_rag(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "What is photosynthesis?"})
        assert result["route"] == ROUTE_LOCAL_RAG
        assert result["subgraph_used"] == "rag_graph"

    def test_missing_tool_falls_back(self, minimal_graph: Any) -> None:
        result = minimal_graph.invoke({"question": "2 + 2 * 3"})
        assert result["subgraph_used"] == "rag_graph"


# ------------------------------------------------------------------ #
#  Math route tests                                                   #
# ------------------------------------------------------------------ #


class TestMathRoute:
    def test_direct_expression(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "100 + 50"})
        assert result["route"] == ROUTE_MATH
        assert "150" in result["answer_text"]

    def test_calculator_result_in_state(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "100 + 50"})
        assert result.get("calculator_result", {}).get("result") == 150

    def test_no_citations(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "100 + 50"})
        assert result.get("citations", []) == []


# ------------------------------------------------------------------ #
#  Web research route tests                                           #
# ------------------------------------------------------------------ #


class TestWebResearchRoute:
    def test_web_search_used_flag(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "What is the latest AI news today?"})
        assert result.get("web_search_used") is True

    def test_web_results_present(self, full_graph: Any) -> None:
        result = full_graph.invoke({"question": "What is the latest AI news today?"})
        assert len(result.get("web_results", [])) > 0


# ------------------------------------------------------------------ #
#  Timeline route tests                                               #
# ------------------------------------------------------------------ #


class TestTimelineRoute:
    def test_timeline_in_state(self, full_graph: Any) -> None:
        result = full_graph.invoke(
            {"question": "Give me a timeline of the Roman Republic"}
        )
        assert len(result.get("timeline", [])) > 0

    def test_citation_validation_present(self, full_graph: Any) -> None:
        result = full_graph.invoke(
            {"question": "Give me a timeline of the Roman Republic"}
        )
        assert "citation_validation" in result


# ------------------------------------------------------------------ #
#  Error handling tests                                               #
# ------------------------------------------------------------------ #


class TestErrorHandling:
    def test_model_failure_doesnt_crash(self) -> None:
        graph = build_supervisor_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FailingModel(),
            routing_mode="heuristic_only",
        )
        result = graph.invoke({"question": "What is photosynthesis?"})
        assert isinstance(result, dict)


# ------------------------------------------------------------------ #
#  Determinism tests                                                  #
# ------------------------------------------------------------------ #


class TestDeterminism:
    def test_deterministic_routing(self) -> None:
        for _ in range(3):
            route, _ = heuristic_route("Calculate 5 + 3")
            assert route == ROUTE_MATH

    def test_deterministic_graph(self, full_graph: Any) -> None:
        r1 = full_graph.invoke({"question": "100 + 50"})
        r2 = full_graph.invoke({"question": "100 + 50"})
        assert r1["route"] == r2["route"]
        assert r1["subgraph_used"] == r2["subgraph_used"]
        assert r1["answer_text"] == r2["answer_text"]


# ------------------------------------------------------------------ #
#  Hybrid routing tests                                               #
# ------------------------------------------------------------------ #


class TestHybridRouting:
    def test_obvious_case_uses_heuristic(self) -> None:
        graph = build_supervisor_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            calculator_tool=FakeCalculatorTool(),
            routing_mode="hybrid",
        )
        result = graph.invoke({"question": "100 + 50"})
        assert result["router_method"] == "heuristic"
        assert result["route"] == ROUTE_MATH

    def test_ambiguous_case_uses_llm(self) -> None:
        graph = build_supervisor_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeRouterModel("local_rag"),
            routing_mode="hybrid",
        )
        result = graph.invoke({"question": "Tell me something interesting about space"})
        assert result["router_method"] == "llm"

    def test_heuristic_only_never_calls_llm(self) -> None:
        graph = build_supervisor_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            routing_mode="heuristic_only",
        )
        result = graph.invoke({"question": "Tell me something interesting about space"})
        assert result["router_method"] == "heuristic"

    def test_llm_only_always_calls_llm(self) -> None:
        graph = build_supervisor_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeRouterModel("local_rag"),
            routing_mode="llm_only",
        )
        result = graph.invoke({"question": "2 + 2"})
        assert result["router_method"] == "llm"


# ------------------------------------------------------------------ #
#  Constants / exports tests                                          #
# ------------------------------------------------------------------ #


class TestConstants:
    def test_valid_routes_is_list(self) -> None:
        assert isinstance(VALID_ROUTES, list)
        assert len(VALID_ROUTES) >= 3

    def test_default_in_valid(self) -> None:
        assert DEFAULT_ROUTE in VALID_ROUTES

    def test_all_route_constants_in_valid(self) -> None:
        for r in [ROUTE_LOCAL_RAG, ROUTE_WEB_RESEARCH, ROUTE_TIMELINE, ROUTE_MATH]:
            assert r in VALID_ROUTES

    def test_ambiguous_not_in_valid(self) -> None:
        assert ROUTE_AMBIGUOUS not in VALID_ROUTES
