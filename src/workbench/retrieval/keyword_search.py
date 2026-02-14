"""
Keyword search adapter backed by LanceDB full-text search (Tantivy/BM25).

Runs a BM25 full-text query on the ``text`` column of the chunks table
and returns results as ``RetrievalResult`` objects.

Satisfies the ``KeywordSearcher`` protocol from
``workbench.core.interfaces``.

Usage:
    from workbench.retrieval.keyword_search import LanceDBKeywordSearcher

    searcher = LanceDBKeywordSearcher(
        db_path=Path("data/indexes/active/lancedb"),
    )
    results = searcher.search(Query(text="French Revolution"), top_k=20)
"""

from __future__ import annotations

import time
from pathlib import Path

import lancedb

from workbench.core.types import Query, RetrievalResult
from workbench.observability.tracing import add_span_attributes, start_span


class LanceDBKeywordSearcher:
    """
    Keyword (BM25) search over a LanceDB chunks table with an FTS index.

    The FTS index must already exist — build it first with
    ``workbench.data_build.index_keyword.build_keyword_index``.

    Args:
        db_path: Path to the LanceDB database directory.
        table_name: Name of the LanceDB table to search.
    """

    def __init__(
        self,
        db_path: Path,
        table_name: str = "chunks",
    ) -> None:
        self.db_path = db_path
        self.table_name = table_name

        self._db = lancedb.connect(str(db_path))
        self._table = self._db.open_table(table_name)

    def search(self, query: Query, top_k: int = 20) -> list[RetrievalResult]:
        """
        Search for chunks matching the query using BM25 full-text search.

        Args:
            query: Query object containing the search text.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievalResult ordered by BM25 score (highest first).
        """
        with start_span(
            "retrieve.keyword",
            attributes={
                "query_text": query.text[:200],
                "top_k": top_k,
                "table_name": self.table_name,
            },
        ):
            t0 = time.time()

            raw_results = (
                self._table.search(query.text, query_type="fts").limit(top_k).to_list()
            )

            results: list[RetrievalResult] = []
            for row in raw_results:
                score = float(row.get("_score", 0.0))

                results.append(
                    RetrievalResult(
                        chunk_id=row["chunk_id"],
                        score=score,
                        source="keyword",
                        metadata={
                            "title": row.get("title", ""),
                            "section": row.get("section", ""),
                            "document_id": row.get("document_id", ""),
                            "text": row.get("text", ""),
                        },
                    )
                )

            duration_ms = (time.time() - t0) * 1000

            add_span_attributes(
                {
                    "results_count": len(results),
                    "duration_ms": round(duration_ms, 1),
                }
            )

            return results

    def __repr__(self) -> str:
        return f"LanceDBKeywordSearcher(db={self.db_path}, table={self.table_name!r})"
