"""
Tests for vector search with a small in-memory LanceDB index.

Creates 5 well-separated chunks, embeds them, writes to a temporary
LanceDB, and verifies that searching for a topic returns the expected
chunk at or near the top.

No external server needed — runs fully locally.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from workbench.core.types import Query
from workbench.data_build.embeddings import SentenceTransformerEmbedder
from workbench.retrieval.vector_search import LanceDBVectorSearcher

# --- Fixtures -----------------------------------------------------------------

# Five chunks with clearly distinct topics so retrieval is unambiguous.
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
def embedder() -> SentenceTransformerEmbedder:
    """Load the embedding model once for all tests in this module."""
    return SentenceTransformerEmbedder(
        model_name="intfloat/multilingual-e5-base",
        device="mps",
        batch_size=8,
    )


@pytest.fixture(scope="module")
def vector_searcher(
    embedder: SentenceTransformerEmbedder,
) -> LanceDBVectorSearcher:
    """
    Build a tiny LanceDB index with 5 chunks and return a searcher.

    Uses a temporary directory that is cleaned up after all module tests.
    """
    import lancedb

    tmp_dir = tempfile.mkdtemp(prefix="test_vector_")
    db_path = Path(tmp_dir) / "lancedb"
    db_path.mkdir(parents=True, exist_ok=True)

    # Embed the sample texts
    texts = [c["text"] for c in SAMPLE_CHUNKS]
    vectors = embedder.embed_texts(texts)

    # Build records
    records = []
    for chunk, vec in zip(SAMPLE_CHUNKS, vectors):
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "text": chunk["text"],
                "title": chunk["title"],
                "section": chunk["section"],
                "vector": vec,
            }
        )

    # Write to LanceDB
    db = lancedb.connect(str(db_path))
    db.create_table("chunks", data=records)

    searcher = LanceDBVectorSearcher(
        db_path=db_path,
        embedder=embedder,
        table_name="chunks",
    )

    yield searcher

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Tests --------------------------------------------------------------------


class TestVectorSearchSmall:
    """Tests for vector search on a small index."""

    def test_search_returns_requested_top_k(
        self, vector_searcher: LanceDBVectorSearcher
    ) -> None:
        """Search should return exactly top_k results (or table size if smaller)."""
        query = Query(text="How do plants make food?")
        results = vector_searcher.search(query, top_k=3)
        assert len(results) == 3

    def test_search_returns_all_when_top_k_exceeds_table(
        self, vector_searcher: LanceDBVectorSearcher
    ) -> None:
        """When top_k > table size, return all rows."""
        query = Query(text="test query")
        results = vector_searcher.search(query, top_k=100)
        assert len(results) == len(SAMPLE_CHUNKS)

    def test_photosynthesis_query_finds_photosynthesis_chunk(
        self, vector_searcher: LanceDBVectorSearcher
    ) -> None:
        """A query about plants and sunlight should rank the photosynthesis chunk highest."""
        query = Query(text="How do green plants use sunlight to produce energy?")
        results = vector_searcher.search(query, top_k=5)

        # The photosynthesis chunk should be the top result
        top_ids = [r.chunk_id for r in results[:2]]
        assert "chunk_photo" in top_ids, (
            f"Expected chunk_photo in top 2, got: {top_ids}"
        )

    def test_gravity_query_finds_gravity_chunk(
        self, vector_searcher: LanceDBVectorSearcher
    ) -> None:
        """A query about gravitational force should rank the gravity chunk highest."""
        query = Query(text="What is the gravitational force between massive objects?")
        results = vector_searcher.search(query, top_k=5)

        top_ids = [r.chunk_id for r in results[:2]]
        assert "chunk_gravity" in top_ids, (
            f"Expected chunk_gravity in top 2, got: {top_ids}"
        )

    def test_programming_query_finds_python_chunk(
        self, vector_searcher: LanceDBVectorSearcher
    ) -> None:
        """A query about coding should rank the Python chunk highest."""
        query = Query(text="What programming language is known for readable syntax?")
        results = vector_searcher.search(query, top_k=5)

        top_ids = [r.chunk_id for r in results[:2]]
        assert "chunk_python" in top_ids, (
            f"Expected chunk_python in top 2, got: {top_ids}"
        )

    def test_results_have_correct_source(
        self, vector_searcher: LanceDBVectorSearcher
    ) -> None:
        """All results should be tagged with source='vector'."""
        query = Query(text="ocean water movement")
        results = vector_searcher.search(query, top_k=3)

        for r in results:
            assert r.source == "vector"

    def test_results_have_positive_scores(
        self, vector_searcher: LanceDBVectorSearcher
    ) -> None:
        """All scores should be positive (our conversion ensures this)."""
        query = Query(text="music composers")
        results = vector_searcher.search(query, top_k=5)

        for r in results:
            assert r.score > 0, f"Score should be positive, got {r.score}"

    def test_results_are_sorted_by_score_descending(
        self, vector_searcher: LanceDBVectorSearcher
    ) -> None:
        """Results should come back sorted by score, highest first."""
        query = Query(text="What is photosynthesis?")
        results = vector_searcher.search(query, top_k=5)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores not sorted descending: {scores}"
        )

    def test_result_metadata_contains_title(
        self, vector_searcher: LanceDBVectorSearcher
    ) -> None:
        """Each result's metadata should include the chunk title."""
        query = Query(text="ocean currents")
        results = vector_searcher.search(query, top_k=3)

        for r in results:
            assert "title" in r.metadata
            assert isinstance(r.metadata["title"], str)
            assert len(r.metadata["title"]) > 0
