"""
Tests for keyword (BM25) search with a small in-memory LanceDB index.

Creates 5 well-separated chunks (same set as the vector search tests),
builds an FTS index, and verifies that searching for distinctive terms
returns the expected chunks.

No external server needed — runs fully locally.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
from workbench.retrieval.keyword_search import LanceDBKeywordSearcher

from workbench.core.types import Query

# Five chunks with clearly distinct vocabulary.
SAMPLE_CHUNKS = [
    {
        "chunk_id": "chunk_photo",
        "document_id": "doc_1",
        "title": "Photosynthesis",
        "section": "",
        "text": (
            "Photosynthesis is the process by which green plants and certain "
            "other organisms transform light energy into chemical energy. "
            "During photosynthesis, chlorophyll absorbs sunlight and uses it "
            "to convert carbon dioxide and water into glucose and oxygen."
        ),
    },
    {
        "chunk_id": "chunk_gravity",
        "document_id": "doc_2",
        "title": "Gravity",
        "section": "",
        "text": (
            "Gravity is a fundamental force of nature that causes objects "
            "with mass to attract one another. Isaac Newton described gravity "
            "as a force, while Albert Einstein described it as a curvature "
            "of spacetime caused by mass and energy."
        ),
    },
    {
        "chunk_id": "chunk_python",
        "document_id": "doc_3",
        "title": "Python Programming",
        "section": "",
        "text": (
            "Python is a high-level, interpreted programming language known "
            "for its clear syntax and readability. It supports multiple "
            "programming paradigms including procedural, object-oriented, "
            "and functional programming."
        ),
    },
    {
        "chunk_id": "chunk_ocean",
        "document_id": "doc_4",
        "title": "Ocean Currents",
        "section": "",
        "text": (
            "Ocean currents are large-scale movements of seawater driven by "
            "wind, temperature differences, salinity, and the rotation of "
            "the Earth. The Gulf Stream is one of the strongest ocean "
            "currents in the Atlantic Ocean."
        ),
    },
    {
        "chunk_id": "chunk_music",
        "document_id": "doc_5",
        "title": "Classical Music",
        "section": "",
        "text": (
            "Classical music is a genre of Western art music that emerged "
            "during the Classical period. Composers such as Mozart, "
            "Beethoven, and Bach are among the most celebrated figures "
            "in the history of classical music."
        ),
    },
]


@pytest.fixture(scope="module")
def keyword_searcher() -> LanceDBKeywordSearcher:
    """
    Build a tiny LanceDB with 5 chunks + FTS index and return a searcher.
    """
    import lancedb

    tmp_dir = tempfile.mkdtemp(prefix="test_keyword_")
    db_path = Path(tmp_dir) / "lancedb"
    db_path.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(db_path))

    # LanceDB needs a vector column even if we only use FTS here.
    # Use a dummy vector so the table schema is consistent.
    dummy_dim = 4
    records = []
    for chunk in SAMPLE_CHUNKS:
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "section": chunk["section"],
                "text": chunk["text"],
                "vector": [0.0] * dummy_dim,
            }
        )

    table = db.create_table("chunks", data=records)

    # Build FTS index on the text column
    table.create_fts_index("text", replace=True)

    searcher = LanceDBKeywordSearcher(
        db_path=db_path,
        table_name="chunks",
    )

    yield searcher

    shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Tests -------------------------------------------------------------------


class TestKeywordSearchSmall:
    """Tests for keyword BM25 search on a small index."""

    def test_search_returns_results(
        self, keyword_searcher: LanceDBKeywordSearcher
    ) -> None:
        """A query for a word present in the corpus should return results."""
        query = Query(text="photosynthesis")
        results = keyword_searcher.search(query, top_k=5)
        assert len(results) >= 1

    def test_photosynthesis_query_finds_photosynthesis_chunk(
        self, keyword_searcher: LanceDBKeywordSearcher
    ) -> None:
        """Searching for 'chlorophyll sunlight' should rank the photo chunk first."""
        query = Query(text="chlorophyll sunlight glucose")
        results = keyword_searcher.search(query, top_k=5)

        top_ids = [r.chunk_id for r in results[:2]]
        assert "chunk_photo" in top_ids, (
            f"Expected chunk_photo in top 2, got: {top_ids}"
        )

    def test_gravity_query_finds_gravity_chunk(
        self, keyword_searcher: LanceDBKeywordSearcher
    ) -> None:
        """Searching for 'Newton Einstein spacetime' should find gravity chunk."""
        query = Query(text="Newton Einstein spacetime")
        results = keyword_searcher.search(query, top_k=5)

        top_ids = [r.chunk_id for r in results[:2]]
        assert "chunk_gravity" in top_ids, (
            f"Expected chunk_gravity in top 2, got: {top_ids}"
        )

    def test_results_have_source_keyword(
        self, keyword_searcher: LanceDBKeywordSearcher
    ) -> None:
        """All results should be tagged with source='keyword'."""
        query = Query(text="programming language")
        results = keyword_searcher.search(query, top_k=3)

        for r in results:
            assert r.source == "keyword"

    def test_results_have_positive_scores(
        self, keyword_searcher: LanceDBKeywordSearcher
    ) -> None:
        """BM25 scores should be positive."""
        query = Query(text="ocean currents Atlantic")
        results = keyword_searcher.search(query, top_k=5)

        for r in results:
            assert r.score > 0, f"Expected positive score, got {r.score}"

    def test_result_metadata_contains_text(
        self, keyword_searcher: LanceDBKeywordSearcher
    ) -> None:
        """Result metadata should include the chunk text for downstream use."""
        query = Query(text="Mozart Beethoven Bach")
        results = keyword_searcher.search(query, top_k=3)

        for r in results:
            assert "text" in r.metadata
            assert len(r.metadata["text"]) > 0

    def test_top_k_limits_results(
        self, keyword_searcher: LanceDBKeywordSearcher
    ) -> None:
        """Should not return more than top_k results."""
        query = Query(text="is")
        results = keyword_searcher.search(query, top_k=2)
        assert len(results) <= 2
