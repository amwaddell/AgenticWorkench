"""
Reranker adapters.

A reranker takes a query and a set of candidate chunks, scores each
(query, chunk_text) pair with a cross-encoder, and returns the
candidates sorted by relevance.

Cross-encoders are much more accurate than bi-encoders for scoring
individual pairs but are too slow to run over the entire index — that's
why we rerank only the small candidate set produced by hybrid retrieval.

Satisfies the ``Reranker`` protocol from ``workbench.core.interfaces``.

Usage:
    from workbench.retrieval.rerankers import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
    results  = reranker.rerank(query, candidates, top_n=5)
"""

from __future__ import annotations

import time

from workbench.core.types import Chunk, Query, RerankResult
from workbench.observability.tracing import add_span_attributes, start_span


class CrossEncoderReranker:
    """
    Reranker backed by a sentence-transformers ``CrossEncoder`` model.

    Args:
        model_name: HuggingFace model ID (e.g. ``"BAAI/bge-reranker-v2-m3"``).
        device: ``"cpu"``, ``"mps"``, or ``"cuda"``.
                ``None`` lets sentence-transformers pick automatically.
        batch_size: Batch size for ``model.predict``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: Query,
        candidates: list[Chunk],
        top_n: int = 5,
    ) -> list[RerankResult]:
        """
        Rerank candidate chunks by cross-encoder relevance.

        Args:
            query: Query object.
            candidates: List of candidate Chunk objects.
            top_n: Number of top results to return.

        Returns:
            List of RerankResult sorted by rerank_score descending.
        """
        if not candidates:
            return []

        with start_span(
            "rerank.cross_encoder",
            attributes={
                "model_name": self.model_name,
                "query_text": query.text[:200],
                "input_count": len(candidates),
                "top_n": top_n,
            },
        ):
            t0 = time.time()

            # Build (query, passage) pairs
            pairs = [(query.text, chunk.text) for chunk in candidates]

            # Score all pairs
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            # Pair each candidate with its score
            scored = list(zip(candidates, scores))

            # Sort by score descending
            scored.sort(key=lambda x: float(x[1]), reverse=True)

            # Build results (take top_n)
            results: list[RerankResult] = []
            for chunk, score in scored[:top_n]:
                results.append(
                    RerankResult(
                        chunk_id=chunk.chunk_id,
                        rerank_score=float(score),
                        original_score=None,
                    )
                )

            duration_ms = (time.time() - t0) * 1000

            add_span_attributes(
                {
                    "output_count": len(results),
                    "duration_ms": round(duration_ms, 1),
                    "top_score": round(float(results[0].rerank_score), 4)
                    if results
                    else 0.0,
                }
            )

            return results

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self.model_name!r})"
