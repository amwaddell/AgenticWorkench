"""
Web search tool: general-purpose internet search.

Supports two providers, selectable via config or constructor arg:

- **duckduckgo** (default) — zero-key, uses the ``duckduckgo_search``
  package (``ddgs``).  Good enough for most workbench experiments.
- **tavily** — higher-quality results, requires an API key via the
  ``TAVILY_API_KEY`` environment variable.

Both providers return a uniform result schema so downstream nodes
(answer_node, etc.) don't need to care which backend was used.

Migrated to ``ToolSpec``: args are validated via Pydantic, and
tracing / logging / metrics hooks are automatic.

Usage (direct)::

    tool = WebSearchTool()                          # DuckDuckGo
    tool = WebSearchTool(provider="tavily")         # Tavily
    result = tool.execute(query="LangGraph tutorial", max_results=5)

Usage (LangGraph)::

    lc_tool = tool.to_langchain_tool()
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from workbench.observability.tracing import add_span_attributes
from workbench.tools.base import ToolSpec

# ------------------------------------------------------------------ #
#  Args schema                                                        #
# ------------------------------------------------------------------ #


class WebSearchArgs(BaseModel):
    """Input schema for the web_search tool."""

    query: str = Field(..., description="The search query.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return.",
    )


# ------------------------------------------------------------------ #
#  Provider back-ends                                                 #
# ------------------------------------------------------------------ #


def _search_duckduckgo(query: str, max_results: int) -> list[dict[str, Any]]:
    """Run a DuckDuckGo text search via the ``duckduckgo_search`` package."""
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise ImportError(
            "duckduckgo-search is required for the DuckDuckGo provider. "
            "Install it with: pip install duckduckgo-search"
        ) from exc

    results: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                    "source": "duckduckgo",
                }
            )
    return results


def _search_tavily(query: str, max_results: int) -> list[dict[str, Any]]:
    """Run a Tavily search.  Requires ``TAVILY_API_KEY`` env var."""
    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "tavily-python is required for the Tavily provider. "
            "Install it with: pip install tavily-python"
        ) from exc

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable is required for the "
            "Tavily provider.  Set it or switch to provider='duckduckgo'."
        )

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=max_results)

    results: list[dict[str, Any]] = []
    for r in response.get("results", []):
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "source": "tavily",
            }
        )
    return results


# Provider dispatch table
_PROVIDERS: dict[str, Any] = {
    "duckduckgo": _search_duckduckgo,
    "tavily": _search_tavily,
}


# ------------------------------------------------------------------ #
#  Tool implementation                                                #
# ------------------------------------------------------------------ #


class WebSearchTool(ToolSpec):
    """
    General-purpose web search tool.

    Args:
        provider: ``"duckduckgo"`` (default, zero-key) or ``"tavily"``
                  (requires ``TAVILY_API_KEY`` env var).
    """

    name: str = "web_search"
    description: str = (
        "Search the internet for up-to-date information. "
        "Returns titles, URLs, and snippets from web pages."
    )
    args_schema = WebSearchArgs

    def __init__(self, provider: str = "duckduckgo") -> None:
        if provider not in _PROVIDERS:
            raise ValueError(
                f"Unknown web search provider {provider!r}. "
                f"Choose from: {sorted(_PROVIDERS)}"
            )
        self.provider = provider
        self._search_fn = _PROVIDERS[provider]

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the web search.

        Args (validated by WebSearchArgs):
            query: The search query.
            max_results: Max results to return.

        Returns:
            Dict with keys: results (list of dicts), count, provider.
        """
        query: str = kwargs["query"]
        max_results: int = kwargs["max_results"]

        if not query.strip():
            return {"results": [], "count": 0, "provider": self.provider}

        results = self._search_fn(query, max_results)

        add_span_attributes(
            {
                "results_count": len(results),
                "provider": self.provider,
            }
        )

        return {
            "results": results,
            "count": len(results),
            "provider": self.provider,
        }

    def __repr__(self) -> str:
        return f"WebSearchTool(provider={self.provider!r})"
