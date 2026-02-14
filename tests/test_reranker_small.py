"""
Tests for the cross-encoder reranker with a small set of candidates.

Loads a real cross-encoder model (can take a few seconds on first run).
Verifies that reranking re-orders candidates sensibly and that outputs
match the expected schema.
"""

from __future__ import annotations

import pytest
from workbench.retrieval.rerankers import CrossEncoderReranker

from workbench.core.types import Chunk, Query

# Three candidates: one highly relevant, one tangential, one irrelevant.
QUERY = Query(text="How do plants convert sunlight into energy?")

CANDIDATES = [
    Chunk(
        chunk_id="chunk_irrelevant",
        document_id="doc_3",
        title="Python Programming",
        section=None,
        text=(
            "Python is a high-level, interpreted programming language known "
            "for its clear syntax and readability."
        ),
        start_offset=0,
        end_offset=100,
        token_count=20,
    ),
    Chunk(
        chunk_id="chunk_tangential",
        document_id="doc_4",
        title="Ecology",
        section=None,
        text=(
            "Ecosystems depend on energy flow from the sun through producers "
            "and consumers in a food web."
        ),
        start_offset=0,
        end_offset=100,
        token_count=18,
    ),
    Chunk(
        chunk_id="chunk_relevant",
        document_id="doc_1",
        title="Photosynthesis",
        section=None,
        text=(
            "Photosynthesis is the process by which green plants and certain "
            "other organisms transform light energy into chemical energy. "
            "During photosynthesis, chlorophyll absorbs sunlight and uses it "
            "to convert carbon dioxide and water into glucose and oxygen."
        ),
        start_offset=0,
        end_offset=250,
        token_count=45,
    ),
]


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    """Load cross-encoder model once for all tests in this module."""
    return CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device=None,  # let sentence-transformers auto-select
    )


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    def test_rerank_returns_correct_count(self, reranker: CrossEncoderReranker) -> None:
        """Should return top_n results."""
        results = reranker.rerank(QUERY, CANDIDATES, top_n=2)
        assert len(results) == 2

    def test_rerank_returns_all_when_top_n_exceeds_candidates(
        self, reranker: CrossEncoderReranker
    ) -> None:
        """When top_n > len(candidates), return all candidates."""
        results = reranker.rerank(QUERY, CANDIDATES, top_n=100)
        assert len(results) == len(CANDIDATES)

    def test_results_sorted_by_rerank_score_descending(
        self, reranker: CrossEncoderReranker
    ) -> None:
        """Output should be sorted by rerank_score, highest first."""
        results = reranker.rerank(QUERY, CANDIDATES, top_n=3)
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores not descending: {scores}"
        )

    def test_relevant_chunk_ranked_first(self, reranker: CrossEncoderReranker) -> None:
        """
        The photosynthesis chunk is clearly most relevant to a question
        about plants converting sunlight — it should be ranked first.
        """
        results = reranker.rerank(QUERY, CANDIDATES, top_n=3)
        assert results[0].chunk_id == "chunk_relevant", (
            f"Expected chunk_relevant first, got {results[0].chunk_id}"
        )

    def test_irrelevant_chunk_ranked_last(self, reranker: CrossEncoderReranker) -> None:
        """
        The Python programming chunk has nothing to do with the query
        and should be ranked last.
        """
        results = reranker.rerank(QUERY, CANDIDATES, top_n=3)
        assert results[-1].chunk_id == "chunk_irrelevant", (
            f"Expected chunk_irrelevant last, got {results[-1].chunk_id}"
        )

    def test_rerank_empty_candidates(self, reranker: CrossEncoderReranker) -> None:
        """Reranking an empty list should return an empty list."""
        results = reranker.rerank(QUERY, [], top_n=5)
        assert results == []

    def test_result_has_chunk_id(self, reranker: CrossEncoderReranker) -> None:
        """Every result should carry the original chunk_id."""
        results = reranker.rerank(QUERY, CANDIDATES, top_n=3)
        result_ids = {r.chunk_id for r in results}
        expected_ids = {c.chunk_id for c in CANDIDATES}
        assert result_ids == expected_ids
