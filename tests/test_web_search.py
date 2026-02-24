"""
Tests for the WebSearchTool (Day 5).

Uses a monkey-patched provider function so tests don't hit the
real DuckDuckGo or Tavily APIs.  Validates:

- Schema enforcement (Pydantic)
- Provider dispatch
- Result shape
- Empty query handling
- Error propagation
- ToolSpec integration (execute, to_langchain_tool)
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from workbench.tools.web_search import (
    _PROVIDERS,
    WebSearchArgs,
    WebSearchTool,
)

# ------------------------------------------------------------------ #
#  Fake provider for tests                                            #
# ------------------------------------------------------------------ #

FAKE_RESULTS = [
    {
        "title": "LangGraph Docs",
        "url": "https://example.com/langgraph",
        "snippet": "LangGraph is a framework for building agentic workflows.",
        "source": "fake",
    },
    {
        "title": "RAG Tutorial",
        "url": "https://example.com/rag",
        "snippet": "How to build retrieval-augmented generation systems.",
        "source": "fake",
    },
    {
        "title": "Python AST",
        "url": "https://example.com/ast",
        "snippet": "Safe evaluation using the ast module.",
        "source": "fake",
    },
]


def _fake_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Fake search provider that returns canned results."""
    if not query.strip():
        return []
    return FAKE_RESULTS[:max_results]


# ------------------------------------------------------------------ #
#  Schema tests                                                       #
# ------------------------------------------------------------------ #


class TestWebSearchSchema:
    """Verify Pydantic schema validation."""

    def test_valid_args(self) -> None:
        args = WebSearchArgs(query="test query")
        assert args.query == "test query"
        assert args.max_results == 5  # default

    def test_custom_max_results(self) -> None:
        args = WebSearchArgs(query="test", max_results=10)
        assert args.max_results == 10

    def test_missing_query_raises(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchArgs()  # type: ignore[call-arg]

    def test_max_results_too_high(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchArgs(query="test", max_results=100)

    def test_max_results_too_low(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchArgs(query="test", max_results=0)


# ------------------------------------------------------------------ #
#  Provider tests                                                     #
# ------------------------------------------------------------------ #


class TestProviderSelection:
    """Verify provider dispatch."""

    def test_default_is_duckduckgo(self) -> None:
        tool = WebSearchTool()
        assert tool.provider == "duckduckgo"

    def test_tavily_provider(self) -> None:
        tool = WebSearchTool(provider="tavily")
        assert tool.provider == "tavily"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown web search provider"):
            WebSearchTool(provider="nonexistent")

    def test_providers_dict_has_both(self) -> None:
        assert "duckduckgo" in _PROVIDERS
        assert "tavily" in _PROVIDERS


# ------------------------------------------------------------------ #
#  Execution tests (with fake provider)                               #
# ------------------------------------------------------------------ #


class TestWebSearchExecution:
    """Test execute() with a monkey-patched provider."""

    @pytest.fixture
    def tool(self) -> WebSearchTool:
        t = WebSearchTool(provider="duckduckgo")
        t._search_fn = _fake_search
        return t

    def test_execute_returns_results(self, tool: WebSearchTool) -> None:
        result = tool.execute(query="LangGraph")
        assert result["count"] > 0
        assert len(result["results"]) > 0

    def test_execute_result_shape(self, tool: WebSearchTool) -> None:
        result = tool.execute(query="LangGraph", max_results=2)
        assert "results" in result
        assert "count" in result
        assert "provider" in result
        assert result["count"] == 2

    def test_each_result_has_required_keys(self, tool: WebSearchTool) -> None:
        result = tool.execute(query="test")
        for r in result["results"]:
            assert "title" in r
            assert "url" in r
            assert "snippet" in r

    def test_empty_query_returns_empty(self, tool: WebSearchTool) -> None:
        result = tool.execute(query="")
        assert result["count"] == 0
        assert result["results"] == []

    def test_whitespace_query_returns_empty(self, tool: WebSearchTool) -> None:
        result = tool.execute(query="   ")
        assert result["count"] == 0

    def test_max_results_limits_output(self, tool: WebSearchTool) -> None:
        result = tool.execute(query="test", max_results=1)
        assert result["count"] == 1
        assert len(result["results"]) == 1

    def test_provider_field_in_result(self, tool: WebSearchTool) -> None:
        result = tool.execute(query="test")
        assert result["provider"] == "duckduckgo"


# ------------------------------------------------------------------ #
#  ToolSpec integration                                               #
# ------------------------------------------------------------------ #


class TestWebSearchToolSpec:
    """Verify ToolSpec base class integration."""

    def test_is_toolspec_subclass(self) -> None:
        from workbench.tools.base import ToolSpec

        assert issubclass(WebSearchTool, ToolSpec)

    def test_has_correct_name(self) -> None:
        tool = WebSearchTool()
        assert tool.name == "web_search"

    def test_has_description(self) -> None:
        tool = WebSearchTool()
        assert len(tool.description) > 10

    def test_repr(self) -> None:
        tool = WebSearchTool()
        assert "WebSearchTool" in repr(tool)
        assert "duckduckgo" in repr(tool)


class TestWebSearchLangChain:
    """Verify to_langchain_tool() works."""

    def test_langchain_tool_name(self) -> None:
        try:
            from langchain_core.tools import BaseTool
        except ImportError:
            pytest.skip("langchain-core not installed")

        tool = WebSearchTool()
        tool._search_fn = _fake_search
        lc = tool.to_langchain_tool()
        assert lc.name == "web_search"

    def test_langchain_tool_invoke(self) -> None:
        try:
            from langchain_core.tools import BaseTool
        except ImportError:
            pytest.skip("langchain-core not installed")

        tool = WebSearchTool()
        tool._search_fn = _fake_search
        lc = tool.to_langchain_tool()
        result_str = lc.invoke({"query": "test", "max_results": 2})
        assert isinstance(result_str, str)
        assert "LangGraph" in result_str
