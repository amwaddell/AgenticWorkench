"""
Component interfaces for the agentic workbench.

These protocols define the contracts that all components must follow.
Using Python's Protocol for structural subtyping (duck typing with type hints).
"""

from typing import Any, Protocol

from workbench.core.types import (
    Chunk,
    ChunkRef,
    ModelResponse,
    Query,
    RerankResult,
    RetrievalResult,
)


class LanguageModel(Protocol):
    """Interface for language model adapters."""

    def generate(
        self,
        messages: list[dict[str, str]],
        **settings: Any,
    ) -> ModelResponse:
        """
        Generate text from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **settings: Model-specific settings (temperature, max_tokens, etc.)

        Returns:
            ModelResponse with generated text and metadata
        """
        ...

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        ...


class Embedder(Protocol):
    """Interface for embedding models."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per input text)
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query (may use special query prefix).

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        ...


class KeywordSearcher(Protocol):
    """Interface for keyword-based search (e.g., BM25)."""

    def search(self, query: Query, top_k: int = 20) -> list[RetrievalResult]:
        """
        Search for chunks using keyword matching.

        Args:
            query: Query object
            top_k: Number of results to return

        Returns:
            List of retrieval results with scores
        """
        ...


class VectorSearcher(Protocol):
    """Interface for vector-based search."""

    def search(self, query: Query, top_k: int = 20) -> list[RetrievalResult]:
        """
        Search for chunks using vector similarity.

        Args:
            query: Query object
            top_k: Number of results to return

        Returns:
            List of retrieval results with scores
        """
        ...


class Reranker(Protocol):
    """Interface for reranking retrieved chunks."""

    def rerank(
        self, query: Query, candidates: list[Chunk], top_n: int = 5
    ) -> list[RerankResult]:
        """
        Rerank candidate chunks based on relevance to query.

        Args:
            query: Query object
            candidates: List of candidate chunks
            top_n: Number of top results to return

        Returns:
            List of reranked results with new scores
        """
        ...


class Chunker(Protocol):
    """Interface for document chunking strategies."""

    def chunk_document(self, document: Any) -> list[Chunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Document to chunk (could be Document type or raw text)

        Returns:
            List of chunks
        """
        ...


class Tool(Protocol):
    """Interface for agent tools."""

    name: str
    description: str

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result as dictionary
        """
        ...


class ContextBuilder(Protocol):
    """Interface for building context from retrieved chunks."""

    def build_context(
        self, chunks: list[Chunk], max_tokens: int
    ) -> tuple[str, list[str]]:
        """
        Build context string from chunks within token budget.

        Args:
            chunks: List of chunks to include
            max_tokens: Maximum tokens allowed

        Returns:
            Tuple of (context_text, included_chunk_ids)
        """
        ...


class Retriever(Protocol):
    """
    Interface for complete retrieval pipeline.

    This can combine keyword + vector + reranking.
    Returns lightweight ``ChunkRef`` objects (not full text).
    Use ``ChunkStore.get()`` to fetch full chunk data when needed.
    """

    def retrieve(self, query: Query, top_k: int = 5) -> list[ChunkRef]:
        """
        Retrieve most relevant chunks for a query.

        Args:
            query: Query object
            top_k: Number of final chunks to return

        Returns:
            List of lightweight ChunkRef objects (id, score, title, snippet).
        """
        ...
