"""
Tests for ChunkStore with a small in-memory LanceDB.

Creates 5 well-separated chunks (same set used in keyword/vector tests),
writes them to a temporary LanceDB, and verifies that:
- get() returns a single Chunk or None
- get_many() returns ordered results, skipping missing IDs
- contains() and count() work correctly
- The returned Chunk objects have the right fields

No external server needed — runs fully locally.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import lancedb
import pytest
from workbench.stores.chunk_store import ChunkStore

from workbench.core.types import Chunk

# --- Fixture data (matches keyword/vector test corpora) ----------------------

SAMPLE_CHUNKS = [
    {
        "chunk_id": "photo_001",
        "document_id": "doc_bio",
        "title": "Photosynthesis",
        "section": "Overview",
        "text": (
            "Photosynthesis is the process by which green plants and certain "
            "other organisms transform light energy into chemical energy. During "
            "photosynthesis, chlorophyll absorbs sunlight and converts carbon "
            "dioxide and water into glucose and oxygen."
        ),
        "start_char": 0,
        "end_char": 230,
    },
    {
        "chunk_id": "gravity_001",
        "document_id": "doc_physics",
        "title": "Gravity",
        "section": "Fundamentals",
        "text": (
            "Gravity is the force by which a planet or other body draws objects "
            "toward its center. Newton described gravity as a force, while "
            "Einstein's general relativity describes it as a curvature of "
            "spacetime caused by mass and energy."
        ),
        "start_char": 0,
        "end_char": 240,
    },
    {
        "chunk_id": "python_001",
        "document_id": "doc_cs",
        "title": "Python Programming",
        "section": "Introduction",
        "text": (
            "Python is a high-level, interpreted programming language known for "
            "its clear syntax and readability. It supports multiple paradigms "
            "including object-oriented, functional, and procedural programming."
        ),
        "start_char": 0,
        "end_char": 210,
    },
    {
        "chunk_id": "rome_001",
        "document_id": "doc_history",
        "title": "Ancient Rome",
        "section": "Republic Era",
        "text": (
            "The Roman Republic was the era of classical Roman civilization "
            "beginning with the overthrow of the Roman Kingdom in 509 BC and "
            "ending with the establishment of the Roman Empire in 27 BC."
        ),
        "start_char": 0,
        "end_char": 200,
    },
    {
        "chunk_id": "music_001",
        "document_id": "doc_arts",
        "title": "Classical Music",
        "section": "Baroque Period",
        "text": (
            "The Baroque period in Western music lasted from approximately 1600 "
            "to 1750. Key composers include Johann Sebastian Bach, George "
            "Frideric Handel, and Antonio Vivaldi."
        ),
        "start_char": 0,
        "end_char": 180,
    },
]


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module")
def chunk_store() -> ChunkStore:
    """Build a tiny LanceDB with 5 chunks and return a ChunkStore."""
    tmp_dir = tempfile.mkdtemp(prefix="test_chunk_store_")
    db_path = Path(tmp_dir) / "test.lancedb"

    db = lancedb.connect(str(db_path))
    db.create_table("chunks", data=SAMPLE_CHUNKS)

    store = ChunkStore(db_path=db_path, table_name="chunks")

    yield store

    shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Tests: get() ------------------------------------------------------------


class TestChunkStoreGet:
    """Tests for single-chunk retrieval."""

    def test_get_existing_chunk(self, chunk_store: ChunkStore) -> None:
        """get() with a valid ID returns a Chunk."""
        chunk = chunk_store.get("photo_001")
        assert chunk is not None
        assert isinstance(chunk, Chunk)

    def test_get_returns_correct_id(self, chunk_store: ChunkStore) -> None:
        """Returned chunk has the requested chunk_id."""
        chunk = chunk_store.get("gravity_001")
        assert chunk is not None
        assert chunk.chunk_id == "gravity_001"

    def test_get_returns_correct_title(self, chunk_store: ChunkStore) -> None:
        """Returned chunk has the correct title."""
        chunk = chunk_store.get("python_001")
        assert chunk is not None
        assert chunk.title == "Python Programming"

    def test_get_returns_correct_section(self, chunk_store: ChunkStore) -> None:
        """Returned chunk has the correct section."""
        chunk = chunk_store.get("rome_001")
        assert chunk is not None
        assert chunk.section == "Republic Era"

    def test_get_returns_text(self, chunk_store: ChunkStore) -> None:
        """Returned chunk has non-empty text."""
        chunk = chunk_store.get("music_001")
        assert chunk is not None
        assert len(chunk.text) > 0
        assert "Baroque" in chunk.text

    def test_get_missing_returns_none(self, chunk_store: ChunkStore) -> None:
        """get() with an unknown ID returns None."""
        assert chunk_store.get("nonexistent_999") is None

    def test_get_empty_string_returns_none(self, chunk_store: ChunkStore) -> None:
        """get() with an empty string returns None."""
        assert chunk_store.get("") is None


# --- Tests: get_many() -------------------------------------------------------


class TestChunkStoreGetMany:
    """Tests for batch chunk retrieval."""

    def test_get_many_returns_all(self, chunk_store: ChunkStore) -> None:
        """get_many() with valid IDs returns all of them."""
        ids = ["photo_001", "gravity_001", "python_001"]
        chunks = chunk_store.get_many(ids)
        assert len(chunks) == 3

    def test_get_many_preserves_order(self, chunk_store: ChunkStore) -> None:
        """Results come back in the same order as the requested IDs."""
        ids = ["music_001", "photo_001", "rome_001"]
        chunks = chunk_store.get_many(ids)
        assert [c.chunk_id for c in chunks] == ids

    def test_get_many_skips_missing(self, chunk_store: ChunkStore) -> None:
        """Missing IDs are silently skipped."""
        ids = ["photo_001", "nonexistent", "gravity_001"]
        chunks = chunk_store.get_many(ids)
        assert len(chunks) == 2
        assert chunks[0].chunk_id == "photo_001"
        assert chunks[1].chunk_id == "gravity_001"

    def test_get_many_empty_list(self, chunk_store: ChunkStore) -> None:
        """get_many() with an empty list returns an empty list."""
        assert chunk_store.get_many([]) == []

    def test_get_many_all_missing(self, chunk_store: ChunkStore) -> None:
        """get_many() where all IDs are unknown returns empty."""
        chunks = chunk_store.get_many(["nope_1", "nope_2"])
        assert chunks == []

    def test_get_many_returns_chunk_objects(self, chunk_store: ChunkStore) -> None:
        """All returned items are Chunk instances."""
        chunks = chunk_store.get_many(["photo_001", "python_001"])
        assert all(isinstance(c, Chunk) for c in chunks)


# --- Tests: contains() and count() ------------------------------------------


class TestChunkStoreMetadata:
    """Tests for contains() and count()."""

    def test_contains_existing(self, chunk_store: ChunkStore) -> None:
        assert chunk_store.contains("photo_001") is True

    def test_contains_missing(self, chunk_store: ChunkStore) -> None:
        assert chunk_store.contains("nonexistent") is False

    def test_count(self, chunk_store: ChunkStore) -> None:
        assert chunk_store.count() == 5


# --- Tests: Chunk field correctness ------------------------------------------


class TestChunkFields:
    """Verify that Chunk objects have the expected fields populated."""

    def test_document_id(self, chunk_store: ChunkStore) -> None:
        chunk = chunk_store.get("photo_001")
        assert chunk is not None
        assert chunk.document_id == "doc_bio"

    def test_start_offset(self, chunk_store: ChunkStore) -> None:
        chunk = chunk_store.get("photo_001")
        assert chunk is not None
        assert chunk.start_offset == 0

    def test_token_count_estimated(self, chunk_store: ChunkStore) -> None:
        """token_count should be a rough estimate (len(text) // 4)."""
        chunk = chunk_store.get("photo_001")
        assert chunk is not None
        assert chunk.token_count > 0


# --- Tests: repr ------------------------------------------------------------


class TestChunkStoreRepr:
    """Test string representation."""

    def test_repr_before_load(self) -> None:
        """repr should work even before data is loaded."""
        tmp_dir = tempfile.mkdtemp(prefix="test_repr_")
        db_path = Path(tmp_dir) / "test.lancedb"
        db = lancedb.connect(str(db_path))
        db.create_table("chunks", data=SAMPLE_CHUNKS[:1])

        store = ChunkStore(db_path=db_path)
        r = repr(store)
        assert "ChunkStore" in r
        assert "not loaded" in r

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_repr_after_load(self, chunk_store: ChunkStore) -> None:
        """repr should show chunk count after data is loaded."""
        # Ensure loaded
        chunk_store.get("photo_001")
        r = repr(chunk_store)
        assert "5" in r
