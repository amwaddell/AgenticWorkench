"""
Agent state: the mutable data bag that flows through the agent loop.

Each step reads state, does work, and writes results back into state.
The state is also the primary source for logging and metrics at the
end of a run.

Usage:
    state = AgentState(question="When did the Roman Republic end?")
    state.retrieved.append(RetrievedChunkSummary(...))
    state.step += 1
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunkSummary(BaseModel):
    """Lightweight summary of a retrieved chunk (before opening)."""

    chunk_id: str
    score: float
    title: str = ""
    snippet: str = Field(default="", description="First N chars of chunk text")


class OpenedChunk(BaseModel):
    """Full chunk data after the agent has 'opened' it."""

    chunk_id: str
    document_id: str = ""
    title: str = ""
    section: str | None = None
    text: str = ""


class AgentState(BaseModel):
    """
    Mutable state for a single agent run.

    The agent loop and policy read/write this object.  At the end of
    the run the final snapshot is logged.
    """

    # Input
    question: str

    # Retrieval stage
    retrieved: list[RetrievedChunkSummary] = Field(default_factory=list)

    # Evidence stage
    opened: list[OpenedChunk] = Field(default_factory=list)

    # Generation stage
    answer_text: str = ""
    citations: list[str] = Field(
        default_factory=list, description="chunk_ids cited in the answer"
    )

    # Bookkeeping
    step: int = 0
    done: bool = False
    error: str | None = None

    # Token tracking (filled after generation)
    tokens_in: int = 0
    tokens_out: int = 0
    model_latency_ms: float = 0.0

    def summary(self) -> dict:
        """Compact dict for logging."""
        return {
            "question": self.question[:200],
            "steps": self.step,
            "retrieved_count": len(self.retrieved),
            "opened_count": len(self.opened),
            "answer_length": len(self.answer_text),
            "citations": self.citations,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "done": self.done,
            "error": self.error,
        }
