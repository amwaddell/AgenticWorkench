"""
ChunkStore: single source of truth for fetching chunk data by ID.

Consolidates the duplicated LanceDB lookup logic that previously
lived in both ``HybridRetriever._lookup_chunks`` and
``OpenChunkTool._ensure_lookup``.

All chunk-fetching now goes through this class, making it easy to
swap backends (LanceDB → SQLite → remote API) without touching
callers.

Usage:
    from workbench.stores.chunk_store import ChunkStore

    store = ChunkStore(db_path=Path("data/indexes/active/lancedb"))
    chunk = store.get("abc123")          # single lookup
    chunks = store.get_many(["a", "b"])  # batch lookup
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import lancedb

from workbench.core.types import Chunk
from workbench.observability.tracing import add_span_attributes, start_span


class ChunkStore:
    """
    Read-only store backed by a LanceDB chunks table.

    Loads the full table into an in-memory dict on first access (lazy)
    for O(1) lookups.  This is fine for datasets that fit in memory
    (up to a few million chunks).

    Args:
        db_path: Path to the LanceDB database directory.
        table_name: Name of the chunks table (default ``"chunks"``).
    """

    def __init__(
        self,
        db_path: Path,
        table_name: str = "chunks",
    ) -> None:
        self.db_path = db_path
        self.table_name = table_name

        self._db = lancedb.connect(str(db_path))
        self._table = self._db.open_table(table_name)

        # Lazy-loaded lookup: chunk_id → row dict
        self._lookup: dict[str, dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> Chunk | None:
        """
        Fetch a single chunk by its ID.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            A ``Chunk`` object, or ``None`` if the ID is not found.
        """
        self._ensure_lookup()
        row = self._lookup.get(chunk_id)  # type: ignore[union-attr]
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        """
        Fetch multiple chunks by ID, preserving the requested order.

        IDs that are not found are silently skipped.

        Args:
            chunk_ids: Ordered list of chunk identifiers.

        Returns:
            List of ``Chunk`` objects in the same order (missing IDs omitted).
        """
        if not chunk_ids:
            return []

        with start_span(
            "chunk_store.get_many",
            attributes={"requested_count": len(chunk_ids)},
        ):
            t0 = time.time()
            self._ensure_lookup()

            chunks: list[Chunk] = []
            for cid in chunk_ids:
                row = self._lookup.get(cid)  # type: ignore[union-attr]
                if row is not None:
                    chunks.append(self._row_to_chunk(row))

            duration_ms = (time.time() - t0) * 1000
            add_span_attributes(
                {
                    "found_count": len(chunks),
                    "duration_ms": round(duration_ms, 1),
                }
            )
            return chunks

    def contains(self, chunk_id: str) -> bool:
        """Check whether a chunk ID exists in the store."""
        self._ensure_lookup()
        return chunk_id in self._lookup  # type: ignore[operator]

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        self._ensure_lookup()
        return len(self._lookup)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_lookup(self) -> None:
        """Load the full table into memory on first access."""
        if self._lookup is not None:
            return

        with start_span("chunk_store.load_index"):
            t0 = time.time()
            df = self._table.to_pandas()

            self._lookup = {}
            for _, row in df.iterrows():
                self._lookup[row["chunk_id"]] = row.to_dict()

            add_span_attributes(
                {
                    "total_chunks": len(self._lookup),
                    "load_ms": round((time.time() - t0) * 1000, 1),
                }
            )

    @staticmethod
    def _row_to_chunk(row: dict[str, Any]) -> Chunk:
        """Convert a LanceDB row dict to a ``Chunk`` object."""
        text = row.get("text", "")
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row.get("document_id", ""),
            title=row.get("title", ""),
            section=row.get("section", None),
            text=text,
            start_offset=row.get("start_char", 0),
            end_offset=row.get("end_char", 0),
            token_count=row.get("token_count", len(text) // 4),
        )

    def __repr__(self) -> str:
        loaded = len(self._lookup) if self._lookup is not None else "not loaded"
        return (
            f"ChunkStore(db={self.db_path}, table={self.table_name!r}, chunks={loaded})"
        )
