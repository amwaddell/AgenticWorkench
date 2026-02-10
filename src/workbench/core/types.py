"""
Core type definitions for the agentic workbench.

These are the data structures that flow through the system.
Using Pydantic for automatic validation and serialization.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single document in the knowledge base."""

    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Full document text")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (url, date, etc.)"
    )


class Chunk(BaseModel):
    """A chunk derived from a document."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    title: str = Field(..., description="Document title (for context)")
    section: str | None = Field(None, description="Section heading if available")
    text: str = Field(..., description="Chunk text content")
    start_offset: int = Field(..., description="Character offset in original document")
    end_offset: int = Field(
        ..., description="End character offset in original document"
    )
    token_count: int = Field(..., description="Approximate token count")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional chunk metadata"
    )


class Query(BaseModel):
    """A user query or question."""

    text: str = Field(..., description="Query text")
    query_id: str | None = Field(None, description="Optional query identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional query metadata"
    )


class RetrievalResult(BaseModel):
    """Result from a retrieval operation (keyword or vector)."""

    chunk_id: str = Field(..., description="ID of retrieved chunk")
    score: float = Field(..., description="Retrieval score")
    source: Literal["keyword", "vector", "hybrid"] = Field(
        ..., description="Source of this result"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional retrieval metadata"
    )


class RerankResult(BaseModel):
    """Result after reranking."""

    chunk_id: str = Field(..., description="ID of chunk")
    rerank_score: float = Field(..., description="Reranker score")
    original_score: float | None = Field(
        None, description="Original retrieval score before reranking"
    )


class ModelResponse(BaseModel):
    """Response from a language model."""

    text: str = Field(..., description="Generated text")
    tokens_in: int = Field(..., description="Input token count")
    tokens_out: int = Field(..., description="Output token count")
    latency_ms: float = Field(..., description="Generation latency in milliseconds")
    finish_reason: str | None = Field(None, description="Reason generation stopped")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional model metadata"
    )


class Answer(BaseModel):
    """Final answer to a user query."""

    query: str = Field(..., description="Original query")
    answer_text: str = Field(..., description="Generated answer")
    citations: list[str] = Field(
        default_factory=list, description="Chunk IDs cited in answer"
    )
    retrieved_chunks: list[str] = Field(
        default_factory=list, description="All chunks used in context"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional answer metadata"
    )


class RunRecord(BaseModel):
    """Record of a single run (for evaluation or production)."""

    run_id: str = Field(..., description="Unique run identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Run timestamp"
    )
    config_snapshot: dict[str, Any] = Field(
        ..., description="Frozen config used for this run"
    )
    run_type: Literal["chatbot", "evaluation", "batch"] = Field(
        ..., description="Type of run"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional run metadata"
    )
    status: Literal["running", "completed", "failed"] = Field(
        default="running", description="Run status"
    )
