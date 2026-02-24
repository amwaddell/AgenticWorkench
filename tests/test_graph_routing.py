"""
Tests for graph routing with web search fallback (Day 5).

Uses fake tools and a fake model — no LanceDB, no model server,
no network calls.  Validates:

- Graph compiles with web_search_tool provided
- Graph still compiles without web_search_tool (backward compat)
- Confidence check routes to web_search when results are low
- Confidence check routes to open when results are sufficient
- Web search results appear in state
- Answer node incorporates web results
- extract_web_citations works
"""

from __future__ import annotations

from typing import Any

import pytest

from workbench.core.types import ModelResponse
from workbench.graphs.rag_graph import (
    build_rag_graph,
    extract_citations,
    extract_web_citations,
)

# ------------------------------------------------------------------ #
#  Fake components                                                    #
# ------------------------------------------------------------------ #

FAKE_CHUNKS = [
    {
        "chunk_id": "chunk_001",
        "score": 0.95,
        "title": "Roman Empire",
        "snippet": "The Roman Republic ended...",
    },
    {
        "chunk_id": "chunk_002",
        "score": 0.88,
        "title": "Julius Caesar",
        "snippet": "Caesar was assassinated...",
    },
    {
        "chunk_id": "chunk_003",
        "score": 0.80,
        "title": "Augustus",
        "snippet": "Augustus became emperor...",
    },
]

FAKE_OPENED = [
    {
        "chunk_id": "chunk_001",
        "document_id": "doc_001",
        "title": "Roman Empire",
        "section": "Fall of the Republic",
        "text": "The Roman Republic ended in 27 BC when Octavian became Augustus.",
    },
    {
        "chunk_id": "chunk_002",
        "document_id": "doc_002",
        "title": "Julius Caesar",
        "section": "Assassination",
        "text": "Julius Caesar was assassinated on the Ides of March, 44 BC.",
    },
]

FAKE_WEB_RESULTS = [
    {
        "title": "Fall of the Roman Republic - Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Fall_of_the_Roman_Republic",
        "snippet": "The Roman Republic fell through a series of civil wars.",
        "source": "fake",
    },
    {
        "title": "Roman Republic | History",
        "url": "https://www.history.com/roman-republic",
        "snippet": "The Republic lasted from 509 BC to 27 BC.",
        "source": "fake",
    },
]


class FakeSearchTool:
    """Search tool returning configurable results."""

    def __init__(self, chunks: list[dict] | None = None) -> None:
        self._chunks = chunks if chunks is not None else FAKE_CHUNKS

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        top_k = kwargs.get("top_k", 5)
        return {"chunks": self._chunks[:top_k], "count": len(self._chunks[:top_k])}


class EmptySearchTool:
    """Search tool that returns no results (triggers web fallback)."""

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"chunks": [], "count": 0}


class SingleResultSearchTool:
    """Search tool that returns just 1 result (below threshold=2)."""

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"chunks": FAKE_CHUNKS[:1], "count": 1}


class FakeOpenChunkTool:
    """Open-chunk tool returning canned data."""

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        chunk_id = kwargs.get("chunk_id", "")
        for c in FAKE_OPENED:
            if c["chunk_id"] == chunk_id:
                return {"found": True, "chunk": c}
        return {"found": False, "chunk": None}


class FakeWebSearchTool:
    """Fake web search tool returning canned results."""

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "results": FAKE_WEB_RESULTS,
            "count": len(FAKE_WEB_RESULTS),
            "provider": "fake",
        }


class FakeModel:
    """Model that returns a canned answer with citations."""

    def __init__(self, answer: str | None = None) -> None:
        self._answer = answer or (
            "The Roman Republic ended in 27 BC [chunk_001]. "
            "Caesar was assassinated in 44 BC [chunk_002]. "
            "See also (https://en.wikipedia.org/wiki/Fall_of_the_Roman_Republic)."
        )

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            text=self._answer,
            tokens_in=100,
            tokens_out=50,
            latency_ms=10.0,
        )


# ------------------------------------------------------------------ #
#  Graph compilation tests                                            #
# ------------------------------------------------------------------ #


class TestGraphCompilationWithWebSearch:
    """Verify graph builds with and without web search."""

    def test_builds_without_web_search(self) -> None:
        """Original Day 3 topology still works."""
        graph = build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
        )
        assert hasattr(graph, "invoke")

    def test_builds_with_web_search(self) -> None:
        """Day 5 topology with web search compiles."""
        graph = build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            web_search_tool=FakeWebSearchTool(),
        )
        assert hasattr(graph, "invoke")

    def test_builds_with_custom_threshold(self) -> None:
        graph = build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            web_search_tool=FakeWebSearchTool(),
            confidence_threshold=5,
        )
        assert hasattr(graph, "invoke")


# ------------------------------------------------------------------ #
#  Confidence routing tests                                           #
# ------------------------------------------------------------------ #


class TestConfidenceRouting:
    """Verify the confidence check routes correctly."""

    def test_sufficient_results_skip_web_search(self) -> None:
        """3 results >= threshold 2: web search NOT triggered."""
        graph = build_rag_graph(
            search_tool=FakeSearchTool(),  # returns 3 results
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            web_search_tool=FakeWebSearchTool(),
            confidence_threshold=2,
        )
        result = graph.invoke({"question": "When did the Roman Republic end?"})
        assert result.get("web_search_used") is False
        assert result.get("web_results", []) == []

    def test_low_results_triggers_web_search(self) -> None:
        """1 result < threshold 2: web search IS triggered."""
        graph = build_rag_graph(
            search_tool=SingleResultSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            web_search_tool=FakeWebSearchTool(),
            confidence_threshold=2,
        )
        result = graph.invoke({"question": "When did the Roman Republic end?"})
        assert result.get("web_search_used") is True
        assert len(result.get("web_results", [])) > 0

    def test_zero_results_triggers_web_search(self) -> None:
        """0 results < threshold 2: web search IS triggered."""
        graph = build_rag_graph(
            search_tool=EmptySearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            web_search_tool=FakeWebSearchTool(),
            confidence_threshold=2,
        )
        result = graph.invoke({"question": "What is dark matter?"})
        assert result.get("web_search_used") is True

    def test_high_threshold_always_triggers(self) -> None:
        """Threshold 100 > any realistic result count."""
        graph = build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            web_search_tool=FakeWebSearchTool(),
            confidence_threshold=100,
        )
        result = graph.invoke({"question": "test"})
        assert result.get("web_search_used") is True

    def test_threshold_1_with_1_result_skips(self) -> None:
        """1 result >= threshold 1: no web search."""
        graph = build_rag_graph(
            search_tool=SingleResultSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            web_search_tool=FakeWebSearchTool(),
            confidence_threshold=1,
        )
        result = graph.invoke({"question": "test"})
        assert result.get("web_search_used") is False


# ------------------------------------------------------------------ #
#  Answer node with web results                                       #
# ------------------------------------------------------------------ #


class TestAnswerWithWebResults:
    """Verify the answer node incorporates web search results."""

    def test_answer_produced_with_web_results(self) -> None:
        graph = build_rag_graph(
            search_tool=EmptySearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            web_search_tool=FakeWebSearchTool(),
            confidence_threshold=2,
        )
        result = graph.invoke({"question": "What is dark matter?"})
        assert result.get("answer_text")

    def test_web_results_in_state(self) -> None:
        graph = build_rag_graph(
            search_tool=EmptySearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            web_search_tool=FakeWebSearchTool(),
            confidence_threshold=2,
        )
        result = graph.invoke({"question": "test"})
        web_results = result.get("web_results", [])
        assert len(web_results) == 2
        assert web_results[0]["title"] == "Fall of the Roman Republic - Wikipedia"


# ------------------------------------------------------------------ #
#  Backward compatibility                                             #
# ------------------------------------------------------------------ #


class TestBackwardCompatibility:
    """Existing Day 3 tests should still pass."""

    @pytest.fixture
    def rag_graph(self) -> Any:
        return build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
        )

    def test_invoke_returns_dict(self, rag_graph: Any) -> None:
        result = rag_graph.invoke({"question": "test?"})
        assert isinstance(result, dict)

    def test_invoke_has_answer_text(self, rag_graph: Any) -> None:
        result = rag_graph.invoke({"question": "test?"})
        assert result.get("answer_text")

    def test_invoke_has_citations(self, rag_graph: Any) -> None:
        result = rag_graph.invoke({"question": "test?"})
        assert "citations" in result

    def test_invoke_has_retrieved(self, rag_graph: Any) -> None:
        result = rag_graph.invoke({"question": "test?"})
        assert "retrieved" in result
        assert len(result["retrieved"]) > 0

    def test_invoke_has_opened(self, rag_graph: Any) -> None:
        result = rag_graph.invoke({"question": "test?"})
        assert "opened" in result

    def test_no_web_fields_without_web_tool(self, rag_graph: Any) -> None:
        """Without web_search_tool, web fields should not be set."""
        result = rag_graph.invoke({"question": "test?"})
        # web_search_used should not be set (or False)
        assert not result.get("web_search_used", False)


# ------------------------------------------------------------------ #
#  Citation extraction tests                                          #
# ------------------------------------------------------------------ #


class TestExtractWebCitations:
    """Test the extract_web_citations utility."""

    def test_finds_urls(self) -> None:
        text = "The answer is here (https://example.com/a) and (https://example.com/b)."
        urls = [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        ]
        cited = extract_web_citations(text, urls)
        assert "https://example.com/a" in cited
        assert "https://example.com/b" in cited
        assert "https://example.com/c" not in cited

    def test_no_urls_found(self) -> None:
        cited = extract_web_citations("no urls here", ["https://example.com"])
        assert cited == []

    def test_empty_text(self) -> None:
        cited = extract_web_citations("", ["https://example.com"])
        assert cited == []

    def test_empty_urls(self) -> None:
        cited = extract_web_citations("some text", [])
        assert cited == []


class TestExtractCitationsStillWorks:
    """Verify the original extract_citations is unchanged."""

    def test_finds_known_ids(self) -> None:
        text = "Answer [chunk_001] and [chunk_002]."
        cited = extract_citations(text, ["chunk_001", "chunk_002", "chunk_003"])
        assert "chunk_001" in cited
        assert "chunk_002" in cited
        assert "chunk_003" not in cited

    def test_empty_text(self) -> None:
        assert extract_citations("", ["chunk_001"]) == []

    def test_empty_known(self) -> None:
        assert extract_citations("text [chunk_001]", []) == []
