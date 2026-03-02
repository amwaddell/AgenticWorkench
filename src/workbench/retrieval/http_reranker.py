"""
HTTP reranker client for remote reranking servers (Infinity / TEI / Jina).

Sends (query, documents) to a ``/rerank`` endpoint and returns
scored results sorted by relevance.

This replaces the heavyweight local ``CrossEncoder`` model load with
a lightweight HTTP client — the model weights live in a separate,
long-lived server process.

Satisfies the ``Reranker`` protocol from ``workbench.core.interfaces``.

Usage::

    from workbench.retrieval.http_reranker import HTTPReranker

    reranker = HTTPReranker(
        base_url="http://127.0.0.1:8082",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    results = reranker.rerank(query, candidates, top_n=5)
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from workbench.core.types import Chunk, Query, RerankResult
from workbench.observability.tracing import add_span_attributes, start_span


class HTTPReranker:
    """
    Reranker client that calls a remote ``/rerank`` endpoint.

    Compatible with Infinity, TEI, and Jina-style reranking APIs.
    Satisfies the ``Reranker`` protocol.

    Args:
        base_url: Base URL of the reranking server
                  (e.g. ``http://127.0.0.1:8082``).
        model_name: Model identifier sent in the request body.
        timeout_seconds: HTTP request timeout.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8082",
        model_name: str = "BAAI/bge-reranker-v2-m3",
        timeout_seconds: float = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._client = httpx.Client(timeout=timeout_seconds)

    # ------------------------------------------------------------------ #
    #  Health check                                                       #
    # ------------------------------------------------------------------ #

    def health_check(self) -> bool:
        """Return True if the server is reachable and healthy."""
        try:
            resp = self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    # ------------------------------------------------------------------ #
    #  Public API  (matches Reranker protocol)                            #
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        query: Query,
        candidates: list[Chunk],
        top_n: int = 5,
    ) -> list[RerankResult]:
        """
        Rerank candidate chunks by sending them to the remote server.

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
            "rerank.http",
            attributes={
                "model_name": self.model_name,
                "query_text": query.text[:200],
                "input_count": len(candidates),
                "top_n": top_n,
                "server": self.base_url,
            },
        ):
            t0 = time.time()

            # Build document list
            documents = [chunk.text for chunk in candidates]

            payload: dict[str, Any] = {
                "model": self.model_name,
                "query": query.text,
                "documents": documents,
                "top_n": top_n,
            }

            resp = self._client.post(
                f"{self.base_url}/rerank",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            # Parse response — Infinity and TEI both return:
            #   { "results": [ {"index": 0, "relevance_score": 0.95}, ... ] }
            raw_results = data.get("results", [])

            # Sort by relevance score descending (server should already
            # do this, but be defensive)
            raw_results.sort(
                key=lambda r: float(r.get("relevance_score", 0)),
                reverse=True,
            )

            results: list[RerankResult] = []
            for item in raw_results[:top_n]:
                idx = item["index"]
                score = float(item.get("relevance_score", 0))
                chunk = candidates[idx]
                results.append(
                    RerankResult(
                        chunk_id=chunk.chunk_id,
                        rerank_score=score,
                        original_score=None,
                    )
                )

            duration_ms = (time.time() - t0) * 1000

            add_span_attributes(
                {
                    "output_count": len(results),
                    "duration_ms": round(duration_ms, 1),
                    "top_score": round(results[0].rerank_score, 4) if results else 0.0,
                }
            )

            return results

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return f"HTTPReranker(base_url={self.base_url!r}, model={self.model_name!r})"
