"""
OpenChunk tool: fetch full chunk text + metadata by chunk_id.

This is the agent's "read more" action — after search returns summaries,
the agent opens individual chunks to read the full evidence.

Satisfies the ``Tool`` protocol from ``workbench.core.interfaces``.

Usage:
    tool = OpenChunkTool(db_path=Path("data/indexes/active/lancedb"))
    result = tool.execute(chunk_id="abc123")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import lancedb

from workbench.agents.state import OpenedChunk
from workbench.observability.tracing import add_span_attributes, start_span


class OpenChunkTool:
    """
    Tool that fetches the full text of a chunk by its chunk_id.

    Args:
        db_path: Path to the LanceDB database directory.
        table_name: Name of the chunks table.
    """

    name: str = "open_chunk"
    description: str = (
        "Open a specific chunk by its ID and return the full text, title, and metadata."
    )

    def __init__(
        self,
        db_path: Path,
        table_name: str = "chunks",
    ) -> None:
        self.db_path = db_path
        self.table_name = table_name

        self._db = lancedb.connect(str(db_path))
        self._table = self._db.open_table(table_name)

        # Build an in-memory lookup on first use (lazy)
        self._lookup: dict[str, dict] | None = None

    def _ensure_lookup(self) -> None:
        """Build the chunk_id → row lookup dict (once)."""
        if self._lookup is not None:
            return
        df = self._table.to_pandas()
        self._lookup = {}
        for _, row in df.iterrows():
            self._lookup[row["chunk_id"]] = row.to_dict()

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Fetch a chunk by ID.

        Kwargs:
            chunk_id (str): The chunk ID to fetch.

        Returns:
            Dict with keys:
                found: bool
                chunk: OpenedChunk dict (if found)
                text_length: int
                duration_ms: float
        """
        chunk_id: str = kwargs.get("chunk_id", "")

        with start_span(
            "tool.open_chunk",
            attributes={"chunk_id": chunk_id},
        ):
            t0 = time.time()

            self._ensure_lookup()

            row = self._lookup.get(chunk_id) if self._lookup else None

            if row is None:
                duration_ms = (time.time() - t0) * 1000
                add_span_attributes(
                    {"found": False, "duration_ms": round(duration_ms, 1)}
                )
                return {
                    "found": False,
                    "chunk": None,
                    "text_length": 0,
                    "duration_ms": round(duration_ms, 1),
                }

            opened = OpenedChunk(
                chunk_id=row["chunk_id"],
                document_id=row.get("document_id", ""),
                title=row.get("title", ""),
                section=row.get("section", None),
                text=row.get("text", ""),
            )

            duration_ms = (time.time() - t0) * 1000

            add_span_attributes(
                {
                    "found": True,
                    "text_length": len(opened.text),
                    "title": opened.title[:100],
                    "duration_ms": round(duration_ms, 1),
                }
            )

            return {
                "found": True,
                "chunk": opened.model_dump(),
                "text_length": len(opened.text),
                "duration_ms": round(duration_ms, 1),
            }

    def __repr__(self) -> str:
        return f"OpenChunkTool(db={self.db_path}, table={self.table_name!r})"
