"""
SearchWikipedia tool: run hybrid retrieval and return chunk summaries.

The retriever returns lightweight ``ChunkRef`` objects (no full text).
This tool converts them to ``RetrievedChunkSummary`` dicts for the
agent state.

Migrated to ``ToolSpec`` (Day 2): args are validated via Pydantic,
and tracing / logging / metrics hooks are automatic.

Usage (direct):
    tool = SearchWikipediaTool(retriever=hybrid_retriever)
    result = tool.execute(query="French Revolution", top_k=5)

Usage (LangGraph):
    lc_tool = tool.to_langchain_tool()
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from workbench.agents.state import RetrievedChunkSummary
from workbench.core.types import Query
from workbench.observability.tracing import add_span_attributes
from workbench.tools.base import ToolSpec

# Snippet length for agent-facing summaries
_SNIPPET_CHARS = 120


# ------------------------------------------------------------------ #
#  Args schema                                                        #
# ------------------------------------------------------------------ #


class SearchWikipediaArgs(BaseModel):
    """Input schema for the search_wikipedia tool."""

    query: str = Field(..., description="The search query text.")
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to retrieve.",
    )


# ------------------------------------------------------------------ #
#  Tool implementation                                                #
# ------------------------------------------------------------------ #


class SearchWikipediaTool(ToolSpec):
    """
    Tool that searches Wikipedia chunks via the hybrid retriever.

    Args:
        retriever: Any object with a ``retrieve(query, top_k)`` method
                   returning a list of ``ChunkRef`` objects.
    """

    name: str = "search_wikipedia"
    description: str = (
        "Search the Wikipedia knowledge base for chunks relevant to a query. "
        "Returns chunk IDs, scores, titles and short snippets."
    )
    args_schema = SearchWikipediaArgs

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the search.

        Args (validated by SearchWikipediaArgs):
            query: The search query text.
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with keys: chunks, count, duration_ms.
        """
        query_text: str = kwargs["query"]
        top_k: int = kwargs["top_k"]

        if not query_text:
            return {"chunks": [], "count": 0}

        query = Query(text=query_text)
        refs = self.retriever.retrieve(query, top_k=top_k)

        summaries: list[RetrievedChunkSummary] = []
        for ref in refs:
            summaries.append(
                RetrievedChunkSummary(
                    chunk_id=ref.chunk_id,
                    score=ref.score,
                    title=ref.title,
                    snippet=ref.snippet[:_SNIPPET_CHARS],
                )
            )

        add_span_attributes({"results_count": len(summaries)})

        return {
            "chunks": [s.model_dump() for s in summaries],
            "count": len(summaries),
        }

    def __repr__(self) -> str:
        return f"SearchWikipediaTool(retriever={self.retriever!r})"
