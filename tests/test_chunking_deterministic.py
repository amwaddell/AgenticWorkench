"""
Tests for chunking determinism.

Running the same chunker on the same input must always produce
identical chunk IDs and identical boundaries.  This is critical
for citation reliability — if chunk IDs drift between runs,
saved citations become invalid.
"""

from workbench.data_build.chunking import (
    FixedSizeChunker,
    ParagraphAwareChunker,
    make_chunk_id,
)

# ---------------------------------------------------------------------------
# Sample text (long enough to produce multiple chunks)
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
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

DOC_ID = "test_doc_42"
TITLE = "Earth"


# ═══════════════════════════════════════════════════════════════════════════
# make_chunk_id
# ═══════════════════════════════════════════════════════════════════════════


class TestMakeChunkId:
    """Tests for the chunk ID hashing function."""

    def test_same_inputs_same_id(self):
        """Identical inputs must produce identical IDs."""
        id_a = make_chunk_id("doc1", 0, 100, "fixed_size", "size=1000")
        id_b = make_chunk_id("doc1", 0, 100, "fixed_size", "size=1000")
        assert id_a == id_b

    def test_different_offsets_different_id(self):
        id_a = make_chunk_id("doc1", 0, 100, "fixed_size", "size=1000")
        id_b = make_chunk_id("doc1", 100, 200, "fixed_size", "size=1000")
        assert id_a != id_b

    def test_different_doc_different_id(self):
        id_a = make_chunk_id("doc1", 0, 100, "fixed_size", "size=1000")
        id_b = make_chunk_id("doc2", 0, 100, "fixed_size", "size=1000")
        assert id_a != id_b

    def test_different_chunker_different_id(self):
        id_a = make_chunk_id("doc1", 0, 100, "fixed_size", "size=1000")
        id_b = make_chunk_id("doc1", 0, 100, "paragraph_aware", "target=1000")
        assert id_a != id_b

    def test_id_is_hex_string(self):
        cid = make_chunk_id("doc1", 0, 100, "test", "settings")
        assert isinstance(cid, str)
        assert len(cid) == 16
        int(cid, 16)  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# FixedSizeChunker determinism
# ═══════════════════════════════════════════════════════════════════════════


class TestFixedSizeChunkerDeterminism:
    """Same input → same output, every time."""

    def test_ids_identical_across_runs(self):
        chunker = FixedSizeChunker(chunk_size=300, overlap=50)
        run_a = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)
        run_b = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)

        assert len(run_a) == len(run_b)
        for a, b in zip(run_a, run_b):
            assert a.chunk_id == b.chunk_id

    def test_boundaries_identical_across_runs(self):
        chunker = FixedSizeChunker(chunk_size=300, overlap=50)
        run_a = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)
        run_b = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)

        for a, b in zip(run_a, run_b):
            assert a.start_char == b.start_char
            assert a.end_char == b.end_char
            assert a.text == b.text

    def test_text_identical_across_runs(self):
        chunker = FixedSizeChunker(chunk_size=300, overlap=50)
        run_a = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)
        run_b = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)

        texts_a = [c.text for c in run_a]
        texts_b = [c.text for c in run_b]
        assert texts_a == texts_b

    def test_different_settings_different_ids(self):
        chunker_a = FixedSizeChunker(chunk_size=300, overlap=50)
        chunker_b = FixedSizeChunker(chunk_size=500, overlap=100)

        chunks_a = chunker_a.chunk(DOC_ID, TITLE, SAMPLE_TEXT)
        chunks_b = chunker_b.chunk(DOC_ID, TITLE, SAMPLE_TEXT)

        ids_a = {c.chunk_id for c in chunks_a}
        ids_b = {c.chunk_id for c in chunks_b}
        # Different settings should produce different IDs
        # (they may partially overlap by coincidence, but not fully)
        assert ids_a != ids_b


# ═══════════════════════════════════════════════════════════════════════════
# ParagraphAwareChunker determinism
# ═══════════════════════════════════════════════════════════════════════════


class TestParagraphAwareChunkerDeterminism:
    """Same input → same output, every time."""

    def test_ids_identical_across_runs(self):
        chunker = ParagraphAwareChunker(target_size=400, max_size=600)
        run_a = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)
        run_b = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)

        assert len(run_a) == len(run_b)
        for a, b in zip(run_a, run_b):
            assert a.chunk_id == b.chunk_id

    def test_boundaries_identical_across_runs(self):
        chunker = ParagraphAwareChunker(target_size=400, max_size=600)
        run_a = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)
        run_b = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)

        for a, b in zip(run_a, run_b):
            assert a.start_char == b.start_char
            assert a.end_char == b.end_char
            assert a.text == b.text

    def test_text_identical_across_runs(self):
        chunker = ParagraphAwareChunker(target_size=400, max_size=600)
        run_a = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)
        run_b = chunker.chunk(DOC_ID, TITLE, SAMPLE_TEXT)

        texts_a = [c.text for c in run_a]
        texts_b = [c.text for c in run_b]
        assert texts_a == texts_b
