"""
Tests for citation numberization.

Covers ``numberize_citations()`` and ``numberize_result()`` which
replace raw ``[chunk_id_hash]`` references with sequential ``[1]``,
``[2]``, etc., ordered by first appearance in the answer text.
"""

from __future__ import annotations

from workbench.prompting.citations import (
    numberize_citations,
    numberize_result,
)

# ------------------------------------------------------------------ #
#  Helpers — reusable test fixtures                                    #
# ------------------------------------------------------------------ #


def _make_chunk(
    chunk_id: str,
    title: str = "Test Article",
    section: str | None = None,
    text: str = "Some chunk text for testing purposes.",
) -> dict:
    """Build a minimal opened-chunk dict."""
    return {
        "chunk_id": chunk_id,
        "title": title,
        "section": section,
        "text": text,
    }


# ------------------------------------------------------------------ #
#  numberize_citations — core logic                                    #
# ------------------------------------------------------------------ #


class TestNumberizeCitations:
    """Test the low-level numberize_citations function."""

    def test_single_citation_replaced_with_one(self):
        """A single chunk_id should become [1]."""
        text = "Rome fell in 476 AD [abc123]."
        chunks = [_make_chunk("abc123", "Fall of Rome")]

        numbered, cmap = numberize_citations(text, chunks)

        assert "[1]" in numbered
        assert "[abc123]" not in numbered
        assert len(cmap) == 1
        assert cmap[0]["number"] == 1
        assert cmap[0]["chunk_id"] == "abc123"
        assert cmap[0]["title"] == "Fall of Rome"

    def test_two_distinct_citations_get_sequential_numbers(self):
        """Two different chunk_ids get [1] and [2] in appearance order."""
        text = "Fact A [aaa111]. Fact B [bbb222]."
        chunks = [
            _make_chunk("aaa111", "Article A"),
            _make_chunk("bbb222", "Article B"),
        ]

        numbered, cmap = numberize_citations(text, chunks)

        assert "[1]" in numbered
        assert "[2]" in numbered
        assert numbered.index("[1]") < numbered.index("[2]")
        assert cmap[0]["chunk_id"] == "aaa111"
        assert cmap[1]["chunk_id"] == "bbb222"

    def test_duplicate_citation_reuses_same_number(self):
        """Same chunk_id appearing twice should get the same number."""
        text = "Claim one [abc123]. Claim two [abc123]."
        chunks = [_make_chunk("abc123")]

        numbered, cmap = numberize_citations(text, chunks)

        assert numbered.count("[1]") == 2
        assert "[2]" not in numbered
        assert len(cmap) == 1

    def test_first_appearance_determines_ordering(self):
        """Number assignment follows first-appearance order, not chunk list order."""
        # bbb appears first in text even though aaa is first in chunks list
        text = "First [bbb222]. Second [aaa111]."
        chunks = [
            _make_chunk("aaa111", "Article A"),
            _make_chunk("bbb222", "Article B"),
        ]

        numbered, cmap = numberize_citations(text, chunks)

        # bbb222 appeared first → gets [1]
        assert cmap[0]["chunk_id"] == "bbb222"
        assert cmap[0]["number"] == 1
        # aaa111 appeared second → gets [2]
        assert cmap[1]["chunk_id"] == "aaa111"
        assert cmap[1]["number"] == 2

    def test_adjacent_citations_both_replaced(self):
        """Back-to-back citations like [aaa][bbb] are both replaced."""
        text = "A major event [aaa111][bbb222] occurred."
        chunks = [
            _make_chunk("aaa111"),
            _make_chunk("bbb222"),
        ]

        numbered, cmap = numberize_citations(text, chunks)

        assert "[1][2]" in numbered
        assert len(cmap) == 2

    def test_hallucinated_id_left_untouched(self):
        """IDs not in opened_chunks are left as-is."""
        text = "Fact [abc123]. Hallucinated [zzz999]."
        chunks = [_make_chunk("abc123")]

        numbered, cmap = numberize_citations(text, chunks)

        assert "[1]" in numbered
        assert "[zzz999]" in numbered  # unchanged
        assert len(cmap) == 1

    def test_empty_text_returns_unchanged(self):
        """Empty answer text passes through with empty map."""
        numbered, cmap = numberize_citations("", [_make_chunk("abc123")])

        assert numbered == ""
        assert cmap == []

    def test_empty_chunks_returns_unchanged(self):
        """No opened chunks → nothing to numberize."""
        text = "Some answer [abc123]."
        numbered, cmap = numberize_citations(text, [])

        assert numbered == text
        assert cmap == []

    def test_no_citations_in_text(self):
        """Answer with no bracket tokens → pass through."""
        text = "A plain answer with no citations."
        chunks = [_make_chunk("abc123")]

        numbered, cmap = numberize_citations(text, chunks)

        assert numbered == text
        assert cmap == []

    def test_snippet_is_truncated(self):
        """Snippet should be ≤~150 chars with ellipsis for long text."""
        long_text = "word " * 100  # 500 chars
        text = "Fact [abc123]."
        chunks = [_make_chunk("abc123", text=long_text)]

        _, cmap = numberize_citations(text, chunks)

        snippet = cmap[0]["snippet"]
        assert len(snippet) <= 160  # 150 + word boundary slack + ellipsis
        assert snippet.endswith("…")

    def test_snippet_short_text_no_ellipsis(self):
        """Short chunk text should appear as-is without ellipsis."""
        text = "Fact [abc123]."
        chunks = [_make_chunk("abc123", text="Short text.")]

        _, cmap = numberize_citations(text, chunks)

        assert cmap[0]["snippet"] == "Short text."
        assert "…" not in cmap[0]["snippet"]

    def test_section_included_in_map(self):
        """Section heading should propagate to citation_map entries."""
        text = "Fact [abc123]."
        chunks = [_make_chunk("abc123", section="Early history")]

        _, cmap = numberize_citations(text, chunks)

        assert cmap[0]["section"] == "Early history"

    def test_section_none_when_absent(self):
        """Missing section should be None in citation_map."""
        text = "Fact [abc123]."
        chunks = [_make_chunk("abc123", section=None)]

        _, cmap = numberize_citations(text, chunks)

        assert cmap[0]["section"] is None

    def test_many_citations_numbered_correctly(self):
        """Stress test: 10 distinct citations should get [1] through [10]."""
        ids = [f"chunk{i:03d}" for i in range(10)]
        text = " ".join(f"Fact {i} [{cid}]." for i, cid in enumerate(ids))
        chunks = [_make_chunk(cid, title=f"Art {i}") for i, cid in enumerate(ids)]

        numbered, cmap = numberize_citations(text, chunks)

        assert len(cmap) == 10
        for i, entry in enumerate(cmap):
            assert entry["number"] == i + 1
            assert f"[{i + 1}]" in numbered

    def test_mixed_duplicates_and_unique(self):
        """Mix of repeated and unique IDs."""
        text = "A [aaa]. B [bbb]. C [aaa]. D [ccc]. E [bbb]."
        chunks = [
            _make_chunk("aaa"),
            _make_chunk("bbb"),
            _make_chunk("ccc"),
        ]

        numbered, cmap = numberize_citations(text, chunks)

        # aaa → [1], bbb → [2], ccc → [3]
        assert len(cmap) == 3
        assert numbered == "A [1]. B [2]. C [1]. D [3]. E [2]."

    def test_chunk_id_not_in_brackets_ignored(self):
        """Bare chunk_id without brackets should NOT be numberized."""
        text = "The id abc123 appears bare. But [abc123] is cited."
        chunks = [_make_chunk("abc123")]

        numbered, cmap = numberize_citations(text, chunks)

        # Only the bracketed one is replaced
        assert "[1]" in numbered
        assert "id abc123 appears" in numbered  # bare id unchanged
        assert len(cmap) == 1


# ------------------------------------------------------------------ #
#  numberize_result — convenience wrapper                              #
# ------------------------------------------------------------------ #


class TestNumberizeResult:
    """Test the result-dict wrapper."""

    def test_mutates_answer_text_and_adds_citation_map(self):
        """Should rewrite answer_text and add citation_map key."""
        result = {
            "answer_text": "Fact [abc123].",
            "opened": [_make_chunk("abc123", "Rome")],
            "citations": ["abc123"],
        }

        returned = numberize_result(result)

        # Returns the same dict
        assert returned is result
        # answer_text rewritten
        assert "[1]" in result["answer_text"]
        assert "[abc123]" not in result["answer_text"]
        # citation_map added
        assert len(result["citation_map"]) == 1
        assert result["citation_map"][0]["title"] == "Rome"
        # original citations list preserved
        assert result["citations"] == ["abc123"]

    def test_no_answer_text(self):
        """Empty answer_text → empty citation_map, no crash."""
        result = {
            "answer_text": "",
            "opened": [_make_chunk("abc123")],
        }

        numberize_result(result)

        assert result["citation_map"] == []

    def test_no_opened_chunks(self):
        """No opened chunks → nothing to numberize."""
        result = {
            "answer_text": "Some text [abc123].",
            "opened": [],
        }

        numberize_result(result)

        assert result["citation_map"] == []
        # answer_text unchanged since no chunks to match
        assert "[abc123]" in result["answer_text"]

    def test_missing_keys_handled(self):
        """Result dict missing answer_text or opened → no crash."""
        result = {}
        numberize_result(result)
        assert result["citation_map"] == []

    def test_preserves_other_result_keys(self):
        """Other keys in the result dict are not disturbed."""
        result = {
            "answer_text": "Fact [abc123].",
            "opened": [_make_chunk("abc123")],
            "tokens_in": 100,
            "tokens_out": 50,
            "web_citations": ["https://example.com"],
            "error": None,
        }

        numberize_result(result)

        assert result["tokens_in"] == 100
        assert result["tokens_out"] == 50
        assert result["web_citations"] == ["https://example.com"]
        assert result["error"] is None


# ------------------------------------------------------------------ #
#  Backward compatibility with existing extract/validate               #
# ------------------------------------------------------------------ #


class TestBackwardCompatibility:
    """Ensure existing citation functions still work unchanged."""

    def test_extract_citations_unchanged(self):
        """extract_citations should still find known IDs in text."""
        from workbench.prompting.citations import extract_citations

        text = "The empire fell [abc123] and then [def456] rose."
        known = ["abc123", "def456", "ghi789"]

        cited = extract_citations(text, known)

        assert "abc123" in cited
        assert "def456" in cited
        assert "ghi789" not in cited

    def test_find_all_bracket_ids_unchanged(self):
        """find_all_bracket_ids should still find all bracketed tokens."""
        from workbench.prompting.citations import find_all_bracket_ids

        text = "Fact [abc123]. More [def456]. Repeat [abc123]."
        ids = find_all_bracket_ids(text)

        assert ids == ["abc123", "def456"]  # unique, first-appearance order

    def test_validate_citations_unchanged(self):
        """validate_citations should still work with RAG_DEFAULT policy."""
        from workbench.prompting.citations import (
            RAG_DEFAULT,
            validate_citations,
        )

        text = "Fact [abc123]."
        result = validate_citations(text, ["abc123"], RAG_DEFAULT)

        assert result.valid is True
        assert "abc123" in result.cited_ids

    def test_policies_still_importable(self):
        """Policy presets should still be importable."""
        from workbench.prompting.citations import (
            RAG_DEFAULT,
            RESEARCHER_STRICT,
            CitationPolicy,
        )

        assert isinstance(RAG_DEFAULT, CitationPolicy)
        assert isinstance(RESEARCHER_STRICT, CitationPolicy)
        assert RESEARCHER_STRICT.min_citations == 1
        assert RAG_DEFAULT.min_citations == 0


class TestLLMSelfNumbering:
    """Test behavior when the LLM outputs its own [1], [2] instead of chunk IDs."""

    def test_llm_numbers_produce_empty_citation_map(self):
        """If LLM uses [1], [2] instead of chunk IDs, citation_map is empty."""
        # The LLM wrote [1] and [2] — these don't match any chunk_id
        text = "Fact one [1]. Fact two [2]."
        chunks = [
            _make_chunk("abc123", "Article A"),
            _make_chunk("def456", "Article B"),
        ]

        numbered, cmap = numberize_citations(text, chunks)

        # [1] and [2] don't match any chunk_id, so nothing to numberize
        assert cmap == []
        # Text is unchanged (no matches to replace)
        assert numbered == text

    def test_numberize_result_empty_map_preserves_opened(self):
        """When citation_map is empty, opened chunks are still in result."""
        result = {
            "answer_text": "Fact [1]. Fact [2].",
            "opened": [
                _make_chunk("abc123", "Article A"),
                _make_chunk("def456", "Article B"),
            ],
        }

        numberize_result(result)

        # citation_map empty because [1], [2] don't match chunk IDs
        assert result["citation_map"] == []
        # But opened chunks are still there for the renderer to use
        assert len(result["opened"]) == 2
