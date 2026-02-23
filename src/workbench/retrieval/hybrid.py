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
    from workbench.stores.chunk_store import ChunkStore

    store = ChunkStore(db_path=Path("data/indexes/active/lancedb"))

    retriever = HybridRetriever(
        keyword_searcher=kw_searcher,
        vector_searcher=vec_searcher,
        reranker=cross_enc_reranker,     # optional
        chunk_store=store,               # needed for reranking
    )
    refs = retriever.retrieve(Query(text="French Revolution"), top_k=5)
    # refs is list[ChunkRef] — lightweight, no full text
"""

from __future__ import annotations

import time
from typing import Any

from workbench.core.types import ChunkRef, Query, RerankResult, RetrievalResult
from workbench.observability.tracing import add_span_attributes, start_span

# Snippet length included in ChunkRef objects
_SNIPPET_CHARS = 200

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
        4.  (optional) Cross-encoder rerank via ChunkStore  →  top *top_k*
        5.  Return ``ChunkRef`` objects (lightweight, no full text)

    Satisfies the ``Retriever`` protocol.

    Args:
        keyword_searcher: KeywordSearcher instance.
        vector_searcher: VectorSearcher instance.
        reranker: Optional Reranker instance (needs ``chunk_store`` too).
        chunk_store: ChunkStore for fetching full text during reranking.
            Required when a reranker is provided.
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
        chunk_store: Any | None = None,
        keyword_top_k: int = 20,
        vector_top_k: int = 20,
        merge_top_n: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self.keyword_searcher = keyword_searcher
        self.vector_searcher = vector_searcher
        self.reranker = reranker
        self.chunk_store = chunk_store
        self.keyword_top_k = keyword_top_k
        self.vector_top_k = vector_top_k
        self.merge_top_n = merge_top_n
        self.rrf_k = rrf_k

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

    def retrieve(self, query: Query, top_k: int = 5) -> list[ChunkRef]:
        """
        Full retrieval pipeline: merge → (rerank) → return ChunkRef objects.

        Satisfies the ``Retriever`` protocol.  Returns lightweight
        ``ChunkRef`` objects — no full chunk text.  The agent must
        explicitly open chunks via ``ChunkStore.get()`` to read
        the full evidence.

        Args:
            query: Query object.
            top_k: Number of final chunks to return.

        Returns:
            List of ChunkRef objects, best first.
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

            # Step 1-3: merge keyword + vector via RRF
            merged = self.merge(query)

            if self.reranker is not None and self.chunk_store is not None:
                # Step 4: rerank — needs full text from ChunkStore
                chunk_ids = [r.chunk_id for r in merged]
                full_chunks = self.chunk_store.get_many(chunk_ids)

                rerank_results: list[RerankResult] = self.reranker.rerank(
                    query,
                    full_chunks,
                    top_n=top_k,
                )

                # Build ChunkRef from rerank results + chunk data
                chunk_map = {c.chunk_id: c for c in full_chunks}
                refs = []
                for rr in rerank_results:
                    c = chunk_map.get(rr.chunk_id)
                    if c is not None:
                        refs.append(
                            ChunkRef(
                                chunk_id=rr.chunk_id,
                                score=rr.rerank_score,
                                title=c.title,
                                section=c.section,
                                snippet=c.text[:_SNIPPET_CHARS].replace("\n", " "),
                            )
                        )
            else:
                # No reranker — build ChunkRef from merged metadata
                refs = self._merged_to_refs(merged[:top_k])

            duration_ms = (time.time() - t0) * 1000

            add_span_attributes(
                {
                    "final_count": len(refs),
                    "total_duration_ms": round(duration_ms, 1),
                }
            )

            return refs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merged_to_refs(merged: list[RetrievalResult]) -> list[ChunkRef]:
        """
        Convert merged RetrievalResult objects to ChunkRef.

        Uses the metadata dict (populated by keyword/vector searchers)
        to fill title, section, and snippet without a DB round-trip.
        """
        refs: list[ChunkRef] = []
        for r in merged:
            meta = r.metadata
            text = meta.get("text", "")
            refs.append(
                ChunkRef(
                    chunk_id=r.chunk_id,
                    score=r.score,
                    title=meta.get("title", ""),
                    section=meta.get("section", None),
                    snippet=text[:_SNIPPET_CHARS].replace("\n", " ") if text else "",
                )
            )
        return refs

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(keyword={self.keyword_searcher!r}, "
            f"vector={self.vector_searcher!r}, "
            f"reranker={self.reranker!r})"
        )
