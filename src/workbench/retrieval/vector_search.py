"""
Vector search adapter backed by LanceDB.

Embeds the query using the configured embedding model, then performs
nearest-neighbour search on the LanceDB index.  Returns results as
``RetrievalResult`` objects ready for downstream fusion or reranking.

Satisfies the ``VectorSearcher`` protocol from
``workbench.core.interfaces``.

Usage:
    from workbench.retrieval.vector_search import LanceDBVectorSearcher

    searcher = LanceDBVectorSearcher(
        db_path=Path("data/indexes/active/lancedb"),
        embedder=my_embedder,
    )
    results = searcher.search(Query(text="What is photosynthesis?"), top_k=5)
"""

from __future__ import annotations

import time
from pathlib import Path

import lancedb

from workbench.core.types import Query, RetrievalResult
from workbench.data_build.embeddings import SentenceTransformerEmbedder
from workbench.observability.tracing import add_span_attributes, start_span


class LanceDBVectorSearcher:
    """
    Vector search over a LanceDB chunks table.

    Args:
        db_path: Path to the LanceDB database directory.
        embedder: Embedder instance for query encoding.
        table_name: Name of the LanceDB table to search.
    """

    def __init__(
        self,
        db_path: Path,
        embedder: SentenceTransformerEmbedder,
        table_name: str = "chunks",
    ) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.table_name = table_name

        # Connect and open the table once
        self._db = lancedb.connect(str(db_path))
        self._table = self._db.open_table(table_name)

    def search(self, query: Query, top_k: int = 20) -> list[RetrievalResult]:
        """
        Search for the most similar chunks to the query.

        Args:
            query: Query object containing the search text.
            top_k: Number of nearest neighbours to return.

        Returns:
            List of RetrievalResult with score and chunk_id,
            ordered by descending similarity (highest first).
        """
        with start_span(
            "retrieve.vector",
            attributes={
                "query_text": query.text[:200],
                "top_k": top_k,
                "table_name": self.table_name,
            },
        ):
            t0 = time.time()

            # Embed the query
            query_vector = self.embedder.embed_query(query.text)

            # LanceDB nearest-neighbour search
            raw_results = self._table.search(query_vector).limit(top_k).to_list()

            # Convert to RetrievalResult
            results: list[RetrievalResult] = []
            for row in raw_results:
                # LanceDB returns a "_distance" column (L2 distance).
                # Convert to a similarity score so higher = better.
                # For cosine distance: similarity ≈ 1 - distance
                # For L2 distance on normalised vectors: same idea.
                distance = row.get("_distance", 0.0)
                score = 1.0 / (1.0 + distance)  # bounded (0, 1]

                results.append(
                    RetrievalResult(
                        chunk_id=row["chunk_id"],
                        score=score,
                        source="vector",
                        metadata={
                            "title": row.get("title", ""),
                            "section": row.get("section", ""),
                            "document_id": row.get("document_id", ""),
                            "distance": distance,
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
        return f"LanceDBVectorSearcher(db={self.db_path}, table={self.table_name!r})"
