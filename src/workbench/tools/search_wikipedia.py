"""
SearchWikipedia tool: run hybrid retrieval and return chunk summaries.

Satisfies the ``Tool`` protocol from ``workbench.core.interfaces``.

Usage:
    tool = SearchWikipediaTool(retriever=hybrid_retriever)
    result = tool.execute(query="French Revolution", top_k=5)
"""

from __future__ import annotations

import time
from typing import Any

from workbench.agents.state import RetrievedChunkSummary
from workbench.core.types import Query
from workbench.observability.tracing import add_span_attributes, start_span

# Snippet length for summaries
_SNIPPET_CHARS = 120


class SearchWikipediaTool:
    """
    Tool that searches Wikipedia chunks via the hybrid retriever.

    Args:
        retriever: Any object with a ``retrieve(query, top_k)`` method
                   returning a list of Chunk objects.
    """

    name: str = "search_wikipedia"
    description: str = (
        "Search the Wikipedia knowledge base for chunks relevant to a query. "
        "Returns chunk IDs, scores, titles and short snippets."
    )

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute a search.

        Kwargs:
            query (str): The search query text.
            top_k (int): Number of chunks to retrieve (default 5).

        Returns:
            Dict with keys:
                chunks: list of RetrievedChunkSummary dicts
                count: number of results
                duration_ms: search latency
        """
        query_text: str = kwargs.get("query", "")
        top_k: int = kwargs.get("top_k", 5)

        if not query_text:
            return {"chunks": [], "count": 0, "duration_ms": 0.0}

        with start_span(
            "tool.search_wikipedia",
            attributes={
                "query_text": query_text[:200],
                "top_k": top_k,
            },
        ):
            t0 = time.time()

            query = Query(text=query_text)
            chunks = self.retriever.retrieve(query, top_k=top_k)

            summaries: list[RetrievedChunkSummary] = []
            for i, chunk in enumerate(chunks):
                summaries.append(
                    RetrievedChunkSummary(
                        chunk_id=chunk.chunk_id,
                        score=1.0 / (i + 1),  # rank-based score
                        title=chunk.title,
                        snippet=chunk.text[:_SNIPPET_CHARS].replace("\n", " "),
                    )
                )

            duration_ms = (time.time() - t0) * 1000

            add_span_attributes(
                {
                    "results_count": len(summaries),
                    "duration_ms": round(duration_ms, 1),
                }
            )

            return {
                "chunks": [s.model_dump() for s in summaries],
                "count": len(summaries),
                "duration_ms": round(duration_ms, 1),
            }

    def __repr__(self) -> str:
        return f"SearchWikipediaTool(retriever={self.retriever!r})"
