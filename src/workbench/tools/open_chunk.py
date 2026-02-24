"""
OpenChunk tool: fetch full chunk text + metadata by chunk_id.

This is the agent's "read more" action — after search returns summaries,
the agent opens individual chunks to read the full evidence.

Delegates all LanceDB access to a shared ``ChunkStore`` instance.

Migrated to ``ToolSpec`` (Day 2): args are validated via Pydantic,
and tracing / logging / metrics hooks are automatic.

Usage (direct):
    from workbench.stores.chunk_store import ChunkStore
    store = ChunkStore(db_path=Path("data/indexes/active/lancedb"))
    tool = OpenChunkTool(chunk_store=store)
    result = tool.execute(chunk_id="abc123")

Usage (LangGraph):
    lc_tool = tool.to_langchain_tool()
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from workbench.agents.state import OpenedChunk
from workbench.observability.tracing import add_span_attributes
from workbench.tools.base import ToolSpec

# ------------------------------------------------------------------ #
#  Args schema                                                        #
# ------------------------------------------------------------------ #


class OpenChunkArgs(BaseModel):
    """Input schema for the open_chunk tool."""

    chunk_id: str = Field(..., description="The chunk ID to fetch.")


# ------------------------------------------------------------------ #
#  Tool implementation                                                #
# ------------------------------------------------------------------ #


class OpenChunkTool(ToolSpec):
    """
    Tool that fetches the full text of a chunk by its chunk_id.

    Args:
        chunk_store: A ``ChunkStore`` instance for looking up chunks.
    """

    name: str = "open_chunk"
    description: str = (
        "Open a specific chunk by its ID and return the full text, title, and metadata."
    )
    args_schema = OpenChunkArgs

    def __init__(self, chunk_store: Any) -> None:
        self.chunk_store = chunk_store

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Fetch a chunk by ID.

        Args (validated by OpenChunkArgs):
            chunk_id: The chunk ID to fetch.

        Returns:
            Dict with keys: found, chunk, text_length.
        """
        chunk_id: str = kwargs["chunk_id"]

        chunk = self.chunk_store.get(chunk_id)

        if chunk is None:
            add_span_attributes({"found": False})
            return {
                "found": False,
                "chunk": None,
                "text_length": 0,
            }

        opened = OpenedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            section=chunk.section,
            text=chunk.text,
        )

        add_span_attributes(
            {
                "found": True,
                "text_length": len(opened.text),
                "title": opened.title[:100],
            }
        )

        return {
            "found": True,
            "chunk": opened.model_dump(),
            "text_length": len(opened.text),
        }

    def __repr__(self) -> str:
        return f"OpenChunkTool(store={self.chunk_store!r})"
