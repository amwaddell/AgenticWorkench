"""
HTTP embedding client for remote embedding servers (Infinity / TEI / OpenAI-compatible).

Sends texts to a ``embeddings`` endpoint and returns vectors.
Handles E5-style prefixes client-side, just like the local
``SentenceTransformerEmbedder``.

This replaces the heavyweight local model load with a lightweight HTTP
client — the model weights live in a separate, long-lived server process.

Satisfies the ``Embedder`` protocol from ``workbench.core.interfaces``.

Usage::

    from workbench.data_build.http_embedder import HTTPEmbedder

    embedder = HTTPEmbedder(
        base_url="http://127.0.0.1:8081",
        model_name="intfloat/multilingual-e5-base",
    )
    vectors = embedder.embed_texts(["hello world", "another sentence"])
    qvec    = embedder.embed_query("search query")
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from workbench.observability.tracing import add_span_attributes, start_span

# Models that use the E5 "query: " / "passage: " prefix convention.
_E5_PREFIX_MODELS = {"e5-base", "e5-large", "e5-small", "multilingual-e5"}


def _is_e5_model(model_name: str) -> bool:
    """Check if a model name looks like an E5 model needing prefixes."""
    name_lower = model_name.lower()
    return any(tag in name_lower for tag in _E5_PREFIX_MODELS)


class HTTPEmbedder:
    """
    Embedding client that calls a remote ``embeddings`` endpoint.

    Compatible with Infinity, TEI, and any OpenAI-compatible embedding
    server.  Satisfies the ``Embedder`` protocol.

    Args:
        base_url: Base URL of the embedding server (e.g. ``http://127.0.0.1:8081``).
        model_name: Model identifier sent in the request body.
        timeout_seconds: HTTP request timeout.
        batch_size: Maximum texts per request.  Larger batches are split
                    automatically.
        query_prefix: Prefix prepended to queries.  Auto-detected for E5
                      models.  Set to ``""`` to disable.
        passage_prefix: Prefix prepended to documents.  Auto-detected for
                        E5 models.  Set to ``""`` to disable.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        model_name: str = "intfloat/multilingual-e5-base",
        timeout_seconds: float = 120,
        batch_size: int = 256,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=timeout_seconds)

        # --- Auto-detect E5 prefixes (same logic as local embedder) ---
        if query_prefix is not None:
            self._query_prefix = query_prefix
        elif _is_e5_model(model_name):
            self._query_prefix = "query: "
        else:
            self._query_prefix = ""

        if passage_prefix is not None:
            self._passage_prefix = passage_prefix
        elif _is_e5_model(model_name):
            self._passage_prefix = "passage: "
        else:
            self._passage_prefix = ""

        # Dimension is discovered on first call
        self.dimension: int | None = None

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
    #  Public API  (matches Embedder protocol)                            #
    # ------------------------------------------------------------------ #

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts (for indexing documents).

        Applies the passage prefix if configured.  Automatically splits
        large batches to stay within ``batch_size``.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []

        # Apply passage prefix
        if self._passage_prefix:
            texts = [self._passage_prefix + t for t in texts]

        with start_span(
            "embedding.http_batch",
            attributes={
                "model_name": self.model_name,
                "batch_size": len(texts),
                "total_chars": sum(len(t) for t in texts),
                "server": self.base_url,
                "passage_prefix": self._passage_prefix or "none",
            },
        ):
            t0 = time.time()

            all_vectors: list[list[float]] = []

            # Split into sub-batches if needed
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                vectors = self._embed_batch(batch)
                all_vectors.extend(vectors)

            duration_ms = (time.time() - t0) * 1000

            if all_vectors and self.dimension is None:
                self.dimension = len(all_vectors[0])

            add_span_attributes(
                {
                    "duration_ms": round(duration_ms, 1),
                    "vectors_produced": len(all_vectors),
                    "dimension": self.dimension or 0,
                }
            )

            return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Applies the query prefix if configured.

        Args:
            query: Query text.

        Returns:
            Embedding vector as a list of floats.
        """
        prefixed = self._query_prefix + query if self._query_prefix else query

        with start_span(
            "embedding.http_query",
            attributes={
                "model_name": self.model_name,
                "query_length": len(query),
                "query_prefix": self._query_prefix or "none",
                "server": self.base_url,
            },
        ):
            vectors = self._embed_batch([prefixed])
            return vectors[0]

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send a single batch to the server and parse the response."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }

        resp = self._client.post(
            f"{self.base_url}/embeddings",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        # OpenAI-compatible response: data[i].embedding
        # Sort by index to guarantee order
        items = sorted(data["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in items]

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
        return (
            f"HTTPEmbedder(base_url={self.base_url!r}, "
            f"model={self.model_name!r}, dim={self.dimension})"
        )
