"""
Tests for chunk boundary correctness.

Validates structural invariants that every chunker must satisfy:
    - No empty chunks
    - Chunk sizes within expected min/max range
    - start_char < end_char for every chunk
    - Chunk text matches the source slice (for fixed-size)
    - All chunkers handle edge cases (empty text, very short text)
"""

import pytest

from workbench.data_build.chunking import (
    BaseChunker,
    FixedSizeChunker,
    ParagraphAwareChunker,
    create_chunker,
)

# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

LONG_TEXT = (
    "Earth is the third planet from the Sun and the only astronomical "
    "object known to harbor life. About 29.2% of Earth's surface is land "
    "consisting of continents and islands. The remaining 70.8% is covered "
    "with water, mostly by oceans, seas, gulfs, and other salt-water bodies, "
    "but also by lakes, rivers, and other freshwater.\n\n"
    "Earth's outer layer is divided into several rigid tectonic plates that "
    "migrate across the surface over many millions of years, while its "
    "interior remains active with a solid iron inner core, a liquid outer "
    "core that generates Earth's magnetic field, and a convective mantle "
    "that drives plate tectonics.\n\n"
    "Earth's atmosphere consists mostly of nitrogen and oxygen. More solar "
    "energy is received by tropical regions than polar regions and is "
    "redistributed by atmospheric and ocean circulation. Greenhouse gases "
    "also play an important role in regulating the surface temperature.\n\n"
    "A region's climate is not only determined by latitude, but also by "
    "elevation and proximity to moderating oceans, among other factors. "
    "Severe weather, such as tropical cyclones, thunderstorms, and heat "
    "waves, occurs in most areas and has a large impact on life.\n\n"
    "Earth's gravity interacts with other objects in space, especially "
    "the Moon, which is Earth's only natural satellite. Earth orbits "
    "around the Sun in about 365.25 days."
)

SHORT_TEXT = "A very short article."
EMPTY_TEXT = ""
SINGLE_PARA = "One paragraph with no blank lines, just a solid block of text that goes on for a while to be somewhat realistic in length for testing purposes."

DOC_ID = "doc_1"
TITLE = "Test"

# Parametrize over both chunkers
ALL_CHUNKERS = [
    FixedSizeChunker(chunk_size=300, overlap=50, min_chunk_size=50),
    ParagraphAwareChunker(target_size=400, max_size=600, min_chunk_size=50),
]


# ═══════════════════════════════════════════════════════════════════════════
# Structural invariants (parametrized over both chunkers)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.name)
class TestChunkBoundaryInvariants:
    """Invariants every chunker must satisfy on normal-length text."""

    def test_produces_chunks(self, chunker: BaseChunker):
        """Long text should produce at least one chunk."""
        chunks = chunker.chunk(DOC_ID, TITLE, LONG_TEXT)
        assert len(chunks) > 0

    def test_no_empty_text(self, chunker: BaseChunker):
        """Every chunk must have non-empty text."""
        for c in chunker.chunk(DOC_ID, TITLE, LONG_TEXT):
            assert len(c.text) > 0, f"Empty chunk: {c.chunk_id}"

    def test_start_less_than_end(self, chunker: BaseChunker):
        """start_char must be strictly less than end_char."""
        for c in chunker.chunk(DOC_ID, TITLE, LONG_TEXT):
            assert c.start_char < c.end_char, (
                f"Invalid offsets: start={c.start_char}, end={c.end_char}"
            )

    def test_start_non_negative(self, chunker: BaseChunker):
        """start_char must be >= 0."""
        for c in chunker.chunk(DOC_ID, TITLE, LONG_TEXT):
            assert c.start_char >= 0

    def test_end_within_document(self, chunker: BaseChunker):
        """end_char must not exceed the document length."""
        for c in chunker.chunk(DOC_ID, TITLE, LONG_TEXT):
            assert c.end_char <= len(LONG_TEXT), (
                f"end_char {c.end_char} exceeds doc length {len(LONG_TEXT)}"
            )

    def test_chunk_ids_unique(self, chunker: BaseChunker):
        """All chunk IDs within a document must be unique."""
        chunks = chunker.chunk(DOC_ID, TITLE, LONG_TEXT)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_metadata_consistent(self, chunker: BaseChunker):
        """Every chunk should carry the correct document_id and title."""
        for c in chunker.chunk(DOC_ID, TITLE, LONG_TEXT):
            assert c.document_id == DOC_ID
            assert c.title == TITLE
            assert c.chunker_name == chunker.name


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.name)
class TestChunkEdgeCases:
    """Edge-case handling for all chunkers."""

    def test_empty_text_returns_no_chunks(self, chunker: BaseChunker):
        chunks = chunker.chunk(DOC_ID, TITLE, EMPTY_TEXT)
        assert chunks == []

    def test_short_text_below_min(self, chunker: BaseChunker):
        """Text shorter than min_chunk_size may produce 0 or 1 chunks."""
        chunks = chunker.chunk(DOC_ID, TITLE, "tiny")
        # Either empty (filtered) or a single chunk — both are valid
        assert len(chunks) <= 1

    def test_single_paragraph(self, chunker: BaseChunker):
        """Single paragraph (no blank lines) should still produce chunks."""
        chunks = chunker.chunk(DOC_ID, TITLE, SINGLE_PARA)
        assert len(chunks) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Fixed-size specific
# ═══════════════════════════════════════════════════════════════════════════


class TestFixedSizeSpecific:
    """Tests specific to the fixed-size chunker's behaviour."""

    def test_chunk_size_respected(self):
        """No chunk should exceed chunk_size."""
        chunker = FixedSizeChunker(chunk_size=300, overlap=50)
        for c in chunker.chunk(DOC_ID, TITLE, LONG_TEXT):
            assert len(c.text) <= 300

    def test_overlap_creates_more_chunks(self):
        """Overlap > 0 should produce more chunks than overlap = 0."""
        no_overlap = FixedSizeChunker(chunk_size=300, overlap=0)
        with_overlap = FixedSizeChunker(chunk_size=300, overlap=100)

        chunks_no = no_overlap.chunk(DOC_ID, TITLE, LONG_TEXT)
        chunks_yes = with_overlap.chunk(DOC_ID, TITLE, LONG_TEXT)
        assert len(chunks_yes) > len(chunks_no)

    def test_overlap_must_be_less_than_size(self):
        """overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_text_matches_source_slice(self):
        """Chunk text should equal the source text at [start_char:end_char]."""
        chunker = FixedSizeChunker(chunk_size=300, overlap=50)
        for c in chunker.chunk(DOC_ID, TITLE, LONG_TEXT):
            assert c.text == LONG_TEXT[c.start_char : c.end_char]


# ═══════════════════════════════════════════════════════════════════════════
# Paragraph-aware specific
# ═══════════════════════════════════════════════════════════════════════════


class TestParagraphAwareSpecific:
    """Tests specific to the paragraph-aware chunker's behaviour."""

    def test_respects_min_chunk_size(self):
        """No chunk should be shorter than min_chunk_size."""
        chunker = ParagraphAwareChunker(
            target_size=400, max_size=600, min_chunk_size=50
        )
        for c in chunker.chunk(DOC_ID, TITLE, LONG_TEXT):
            assert len(c.text) >= 50, f"Chunk too short: {len(c.text)} chars"

    def test_multiple_chunks_from_long_text(self):
        """Long text with multiple paragraphs should produce multiple chunks."""
        chunker = ParagraphAwareChunker(target_size=300, max_size=500)
        chunks = chunker.chunk(DOC_ID, TITLE, LONG_TEXT)
        assert len(chunks) > 1


# ═══════════════════════════════════════════════════════════════════════════
# create_chunker factory
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateChunker:
    """Tests for the factory function."""

    def test_creates_fixed_size(self):
        c = create_chunker("fixed_size", chunk_size=500)
        assert isinstance(c, FixedSizeChunker)
        assert c.chunk_size == 500

    def test_creates_paragraph_aware(self):
        c = create_chunker("paragraph_aware", target_size=800)
        assert isinstance(c, ParagraphAwareChunker)
        assert c.target_size == 800

    def test_unknown_chunker_raises(self):
        with pytest.raises(KeyError):
            create_chunker("nonexistent_chunker")
