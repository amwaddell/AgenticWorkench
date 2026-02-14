"""
Hybrid retrieval: merge keyword and vector results with Reciprocal Rank Fusion.

RRF is a simple, effective fusion method that does not require score
calibration between rankers.  For each result the RRF score is::

    rrf_score = Σ  1 / (k + rank_i)

where *k* is a constant (default 60) and *rank_i* is the 1-based rank
from each contributing list.

This module provides:

* ``reciprocal_rank_fusion`` — pure function, easy to unit-test.
* ``HybridRetriever``       — wires keyword + vector searchers, optional
  reranker, and satisfies the ``Retriever`` protocol.

Usage:
    from workbench.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(
        keyword_searcher=kw_searcher,
        vector_searcher=vec_searcher,
        reranker=cross_enc_reranker,     # optional
        db_path=Path("data/indexes/active/lancedb"),
    )
    chunks = retriever.retrieve(Query(text="French Revolution"), top_k=5)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import lancedb

from workbench.core.types import Chunk, Query, RerankResult, RetrievalResult
from workbench.observability.tracing import add_span_attributes, start_span

# ---------------------------------------------------------------------------
# Pure fusion function
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievalResult],
    k: int = 60,
    top_n: int = 50,
) -> list[RetrievalResult]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        *ranked_lists: One or more ranked result lists (best-first order).
        k: RRF constant — higher values dampen rank differences.
            60 is the standard value from the original RRF paper.
        top_n: Maximum number of merged results to return.

    Returns:
        Merged list of RetrievalResult sorted by RRF score (highest first).
        Each result has ``source="hybrid"`` and the RRF score.
    """
    # Accumulate RRF scores per chunk_id
    rrf_scores: dict[str, float] = {}
    # Keep the best metadata seen for each chunk_id
    best_meta: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank_0, result in enumerate(ranked_list):
            rank_1 = rank_0 + 1  # 1-based
            rrf = 1.0 / (k + rank_1)
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + rrf

            # Keep metadata from the first list that provides it
            if result.chunk_id not in best_meta:
                best_meta[result.chunk_id] = dict(result.metadata)

    # Sort by descending RRF score, then by chunk_id for deterministic tie-breaking
    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda cid: (-rrf_scores[cid], cid),
    )

    merged: list[RetrievalResult] = []
    for cid in sorted_ids[:top_n]:
        merged.append(
            RetrievalResult(
                chunk_id=cid,
                score=rrf_scores[cid],
                source="hybrid",
                metadata=best_meta.get(cid, {}),
            )
        )

    return merged


# ---------------------------------------------------------------------------
# HybridRetriever — full pipeline
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Retriever that fuses keyword + vector search and optionally reranks.

    Pipeline:
        1.  Keyword search  →  top *keyword_top_k* results
        2.  Vector search   →  top *vector_top_k* results
        3.  RRF merge       →  top *merge_top_n* candidates
        4.  (optional) Cross-encoder rerank  →  top *top_k* results
        5.  Return Chunk objects for the final candidates

    Satisfies the ``Retriever`` protocol.

    Args:
        keyword_searcher: KeywordSearcher instance.
        vector_searcher: VectorSearcher instance.
        reranker: Optional Reranker instance.
        db_path: LanceDB path — used to look up full chunk data for reranking.
        table_name: LanceDB table name.
        keyword_top_k: How many keyword results to fetch before merge.
        vector_top_k: How many vector results to fetch before merge.
        merge_top_n: How many candidates to keep after RRF merge.
        rrf_k: RRF constant (default 60).
    """

    def __init__(
        self,
        keyword_searcher: Any,
        vector_searcher: Any,
        reranker: Any | None = None,
        db_path: Path | None = None,
        table_name: str = "chunks",
        keyword_top_k: int = 20,
        vector_top_k: int = 20,
        merge_top_n: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self.keyword_searcher = keyword_searcher
        self.vector_searcher = vector_searcher
        self.reranker = reranker
        self.table_name = table_name
        self.keyword_top_k = keyword_top_k
        self.vector_top_k = vector_top_k
        self.merge_top_n = merge_top_n
        self.rrf_k = rrf_k

        # LanceDB connection for chunk lookups
        self._db = None
        self._table = None
        if db_path is not None:
            self._db = lancedb.connect(str(db_path))
            self._table = self._db.open_table(table_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(self, query: Query) -> list[RetrievalResult]:
        """
        Run keyword + vector search and merge with RRF.

        This is the fusion step only — no reranking.

        Args:
            query: Query object.

        Returns:
            Merged RetrievalResult list.
        """
        with start_span(
            "retrieve.hybrid_merge",
            attributes={"query_text": query.text[:200]},
        ):
            t0 = time.time()

            kw_results = self.keyword_searcher.search(query, top_k=self.keyword_top_k)
            vec_results = self.vector_searcher.search(query, top_k=self.vector_top_k)

            merged = reciprocal_rank_fusion(
                kw_results,
                vec_results,
                k=self.rrf_k,
                top_n=self.merge_top_n,
            )

            duration_ms = (time.time() - t0) * 1000

            add_span_attributes(
                {
                    "keyword_count": len(kw_results),
                    "vector_count": len(vec_results),
                    "merged_count": len(merged),
                    "duration_ms": round(duration_ms, 1),
                }
            )

            return merged

    def retrieve(self, query: Query, top_k: int = 5) -> list[Chunk]:
        """
        Full retrieval pipeline: merge → (rerank) → return Chunk objects.

        Satisfies the ``Retriever`` protocol.

        Args:
            query: Query object.
            top_k: Number of final chunks to return.

        Returns:
            List of Chunk objects, best first.
        """
        with start_span(
            "retrieve.hybrid_pipeline",
            attributes={
                "query_text": query.text[:200],
                "top_k": top_k,
                "reranker_enabled": self.reranker is not None,
            },
        ):
            t0 = time.time()

            # Step 1-3: merge
            merged = self.merge(query)

            # Look up full chunk data for merged candidates
            chunks = self._lookup_chunks([r.chunk_id for r in merged])

            if self.reranker is not None and chunks:
                # Step 4: rerank
                rerank_results: list[RerankResult] = self.reranker.rerank(
                    query,
                    chunks,
                    top_n=top_k,
                )
                # Build a map of chunk_id → Chunk for fast lookup
                chunk_map = {c.chunk_id: c for c in chunks}
                final_chunks = [
                    chunk_map[rr.chunk_id]
                    for rr in rerank_results
                    if rr.chunk_id in chunk_map
                ]
            else:
                # No reranker — just take top_k from merged order
                final_chunks = chunks[:top_k]

            duration_ms = (time.time() - t0) * 1000

            add_span_attributes(
                {
                    "final_count": len(final_chunks),
                    "total_duration_ms": round(duration_ms, 1),
                }
            )

            return final_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """
        Fetch full chunk data from LanceDB by chunk_id.

        Args:
            chunk_ids: Ordered list of chunk_ids to look up.

        Returns:
            List of Chunk objects in the same order (missing ids skipped).
        """
        if not chunk_ids or self._table is None:
            return []

        # Query LanceDB for all matching chunk_ids in one go.
        # We use a filter expression to fetch them.
        id_set = set(chunk_ids)

        # LanceDB to_pandas then filter is reliable across versions
        all_data = self._table.to_pandas()
        matched = all_data[all_data["chunk_id"].isin(id_set)]

        # Build a lookup dict
        row_map: dict[str, dict] = {}
        for _, row in matched.iterrows():
            row_map[row["chunk_id"]] = row.to_dict()

        # Preserve the requested order
        chunks: list[Chunk] = []
        for cid in chunk_ids:
            if cid not in row_map:
                continue
            r = row_map[cid]
            chunks.append(
                Chunk(
                    chunk_id=r["chunk_id"],
                    document_id=r.get("document_id", ""),
                    title=r.get("title", ""),
                    section=r.get("section", None),
                    text=r.get("text", ""),
                    start_offset=0,  # not stored in index
                    end_offset=0,  # not stored in index
                    token_count=len(r.get("text", "")) // 4,  # rough estimate
                )
            )

        return chunks

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(keyword={self.keyword_searcher!r}, "
            f"vector={self.vector_searcher!r}, "
            f"reranker={self.reranker!r})"
        )
