"""
Embedding adapter using sentence-transformers.

Loads a HuggingFace sentence-transformer model once and provides:
    - embed_texts(texts)  → list of vectors  (for indexing documents)
    - embed_query(query)  → single vector    (for search queries)

Some models require different prefixes for queries vs documents:
    - E5 models: "query: " for queries, "passage: " for documents
    - Other models may use prompt_name-based routing

This adapter handles prefixes automatically based on model config
or explicit configuration.

Usage:
    from workbench.data_build.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(
        model_name="intfloat/multilingual-e5-base",
        device="mps",
    )
    vectors = embedder.embed_texts(["hello world", "another sentence"])
    qvec    = embedder.embed_query("search query")
"""

from __future__ import annotations

import time

import numpy as np

from workbench.observability.tracing import add_span_attributes, start_span

# Models that use the E5 "query: " / "passage: " prefix convention.
_E5_PREFIX_MODELS = {"e5-base", "e5-large", "e5-small", "multilingual-e5"}


def _is_e5_model(model_name: str) -> bool:
    """Check if a model name looks like an E5 model needing prefixes."""
    name_lower = model_name.lower()
    return any(tag in name_lower for tag in _E5_PREFIX_MODELS)


class SentenceTransformerEmbedder:
    """
    Embedder backed by a sentence-transformers model.

    Satisfies the ``Embedder`` protocol defined in
    ``workbench.core.interfaces``.

    Args:
        model_name: HuggingFace model ID or local path.
        device: ``"cpu"``, ``"mps"`` (Apple Silicon GPU), or ``"cuda"``.
                ``None`` lets sentence-transformers pick automatically.
        batch_size: Internal encode batch size (passed to model.encode).
        query_prefix: Prefix prepended to queries.  Auto-detected for
                      E5 models (``"query: "``).  Set to ``""`` to disable.
        passage_prefix: Prefix prepended to documents.  Auto-detected for
                        E5 models (``"passage: "``).  Set to ``""`` to disable.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str | None = "mps",
        batch_size: int = 32,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension: int = self._model.get_sentence_embedding_dimension()

        # --- Auto-detect prefixes for E5 models ---
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

        self.device = str(self._model.device)

    # ------------------------------------------------------------------
    # Public API  (matches Embedder protocol)
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts (for indexing documents).

        Applies the passage prefix if configured (e.g. ``"passage: "``
        for E5 models).

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
            "embedding.batch",
            attributes={
                "model_name": self.model_name,
                "batch_size": len(texts),
                "total_chars": sum(len(t) for t in texts),
                "device": self.device,
                "passage_prefix": self._passage_prefix or "none",
            },
        ):
            t0 = time.time()
            embeddings: np.ndarray = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            duration_ms = (time.time() - t0) * 1000

            add_span_attributes(
                {
                    "duration_ms": round(duration_ms, 1),
                    "vectors_produced": len(embeddings),
                    "dimension": self.dimension,
                }
            )

            return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Applies the query prefix if configured (e.g. ``"query: "``
        for E5 models).

        Args:
            query: Query text.

        Returns:
            Embedding vector as a list of floats.
        """
        # Apply query prefix
        prefixed = self._query_prefix + query if self._query_prefix else query

        with start_span(
            "embedding.query",
            attributes={
                "model_name": self.model_name,
                "query_length": len(query),
                "query_prefix": self._query_prefix or "none",
                "device": self.device,
            },
        ):
            vec: np.ndarray = self._model.encode(
                [prefixed],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vec[0].tolist()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SentenceTransformerEmbedder(model={self.model_name!r}, "
            f"dim={self.dimension}, device={self.device!r}, "
            f"query_prefix={self._query_prefix!r}, "
            f"passage_prefix={self._passage_prefix!r})"
        )
