"""
Tests for Reciprocal Rank Fusion (RRF) merge logic.

These are pure unit tests — no LanceDB, no models, no I/O.
They verify that the fusion function is deterministic and produces
the expected ordering.
"""

from __future__ import annotations

from workbench.retrieval.hybrid import reciprocal_rank_fusion

from workbench.core.types import RetrievalResult


def _make_result(chunk_id: str, score: float, source: str) -> RetrievalResult:
    """Helper to build a RetrievalResult quickly."""
    return RetrievalResult(chunk_id=chunk_id, score=score, source=source)


# --- Two simple ranked lists for most tests --------------------------------

LIST_KEYWORD = [
    _make_result("A", 10.0, "keyword"),
    _make_result("B", 8.0, "keyword"),
    _make_result("C", 5.0, "keyword"),
]

LIST_VECTOR = [
    _make_result("B", 0.95, "vector"),
    _make_result("D", 0.90, "vector"),
    _make_result("A", 0.80, "vector"),
]


class TestReciprocalRankFusion:
    """Tests for the reciprocal_rank_fusion function."""

    def test_returns_retrieval_results(self) -> None:
        """Output should be a list of RetrievalResult objects."""
        merged = reciprocal_rank_fusion(LIST_KEYWORD, LIST_VECTOR)
        assert all(isinstance(r, RetrievalResult) for r in merged)

    def test_all_results_have_hybrid_source(self) -> None:
        """Merged results should all have source='hybrid'."""
        merged = reciprocal_rank_fusion(LIST_KEYWORD, LIST_VECTOR)
        for r in merged:
            assert r.source == "hybrid"

    def test_union_of_chunk_ids(self) -> None:
        """Merged list should contain the union of both input lists."""
        merged = reciprocal_rank_fusion(LIST_KEYWORD, LIST_VECTOR)
        merged_ids = {r.chunk_id for r in merged}
        assert merged_ids == {"A", "B", "C", "D"}

    def test_b_ranked_first(self) -> None:
        """
        'B' appears in both lists (rank 2 keyword, rank 1 vector),
        so it should get the highest RRF score and be ranked first.
        """
        merged = reciprocal_rank_fusion(LIST_KEYWORD, LIST_VECTOR, k=60)
        assert merged[0].chunk_id == "B"

    def test_a_ranked_second(self) -> None:
        """
        'A' also appears in both lists (rank 1 keyword, rank 3 vector),
        so it should be ranked second after B.
        """
        merged = reciprocal_rank_fusion(LIST_KEYWORD, LIST_VECTOR, k=60)
        assert merged[1].chunk_id == "A"

    def test_scores_are_descending(self) -> None:
        """Merged results should be sorted by RRF score, highest first."""
        merged = reciprocal_rank_fusion(LIST_KEYWORD, LIST_VECTOR)
        scores = [r.score for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_deterministic_across_calls(self) -> None:
        """Same inputs should always produce the same output."""
        m1 = reciprocal_rank_fusion(LIST_KEYWORD, LIST_VECTOR, k=60, top_n=10)
        m2 = reciprocal_rank_fusion(LIST_KEYWORD, LIST_VECTOR, k=60, top_n=10)

        ids_1 = [r.chunk_id for r in m1]
        ids_2 = [r.chunk_id for r in m2]
        assert ids_1 == ids_2

        scores_1 = [r.score for r in m1]
        scores_2 = [r.score for r in m2]
        assert scores_1 == scores_2

    def test_top_n_limits_output(self) -> None:
        """top_n should cap the number of returned results."""
        merged = reciprocal_rank_fusion(LIST_KEYWORD, LIST_VECTOR, top_n=2)
        assert len(merged) == 2

    def test_single_list(self) -> None:
        """Fusion with a single list should preserve the original order."""
        merged = reciprocal_rank_fusion(LIST_KEYWORD, k=60)
        ids = [r.chunk_id for r in merged]
        assert ids == ["A", "B", "C"]

    def test_empty_lists(self) -> None:
        """Fusing two empty lists should return an empty list."""
        merged = reciprocal_rank_fusion([], [])
        assert merged == []

    def test_one_empty_one_populated(self) -> None:
        """Fusing an empty list with a populated one should return the populated items."""
        merged = reciprocal_rank_fusion([], LIST_VECTOR)
        assert len(merged) == len(LIST_VECTOR)

    def test_k_parameter_affects_scores(self) -> None:
        """Changing k should change the absolute scores (but not the order)."""
        m_low_k = reciprocal_rank_fusion(LIST_KEYWORD, LIST_VECTOR, k=1)
        m_high_k = reciprocal_rank_fusion(LIST_KEYWORD, LIST_VECTOR, k=100)

        # Same order
        assert [r.chunk_id for r in m_low_k] == [r.chunk_id for r in m_high_k]

        # Different absolute scores (lower k => higher individual RRF contributions)
        assert m_low_k[0].score > m_high_k[0].score

    def test_ties_broken_by_chunk_id(self) -> None:
        """
        When two chunk_ids have identical RRF scores, they should be
        ordered lexicographically by chunk_id for determinism.
        """
        # Two lists where X and Y appear at the same rank in exactly one list each
        list_a = [_make_result("Y", 1.0, "keyword")]
        list_b = [_make_result("X", 1.0, "vector")]

        merged = reciprocal_rank_fusion(list_a, list_b, k=60)
        ids = [r.chunk_id for r in merged]
        # Both have RRF score = 1/(60+1).  X < Y lexicographically.
        assert ids == ["X", "Y"]
