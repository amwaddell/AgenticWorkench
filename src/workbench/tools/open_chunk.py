"""
OpenChunk tool: fetch full chunk text + metadata by chunk_id.

This is the agent's "read more" action — after search returns summaries,
the agent opens individual chunks to read the full evidence.

Delegates all LanceDB access to a shared ``ChunkStore`` instance,
eliminating the duplicated lookup logic that previously lived here.

Satisfies the ``Tool`` protocol from ``workbench.core.interfaces``.

Usage:
    from workbench.stores.chunk_store import ChunkStore

    store = ChunkStore(db_path=Path("data/indexes/active/lancedb"))
    tool = OpenChunkTool(chunk_store=store)
    result = tool.execute(chunk_id="abc123")
"""

from __future__ import annotations

import time
from typing import Any

from workbench.agents.state import OpenedChunk
from workbench.observability.tracing import add_span_attributes, start_span


class OpenChunkTool:
    """
    Tool that fetches the full text of a chunk by its chunk_id.

    Args:
        chunk_store: A ``ChunkStore`` instance for looking up chunks.
    """

    name: str = "open_chunk"
    description: str = (
        "Open a specific chunk by its ID and return the full text, title, and metadata."
    )

    def __init__(self, chunk_store: Any) -> None:
        self.chunk_store = chunk_store

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

            chunk = self.chunk_store.get(chunk_id)

            if chunk is None:
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
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                section=chunk.section,
                text=chunk.text,
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
        return f"OpenChunkTool(store={self.chunk_store!r})"
