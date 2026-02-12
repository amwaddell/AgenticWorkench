"""
Chunking strategies for splitting articles into retrieval-ready chunks.

Provides two chunkers:
    - FixedSizeChunker:     splits on character boundaries with overlap
    - ParagraphAwareChunker: splits on paragraph (blank-line) boundaries,
                             merging small paragraphs up to a target size

Both produce stable, deterministic chunk IDs derived from a hash of the
document id, character offsets, chunker name, and chunker settings.

Usage:
    from workbench.data_build.chunking import (
        FixedSizeChunker,
        ParagraphAwareChunker,
        build_chunks_parquet,
    )

    chunker = ParagraphAwareChunker(target_size=1000, max_size=1500)
    chunks  = chunker.chunk(doc_id="12", title="Earth", text="...")

    # Or build the full dataset:
    summary = build_chunks_parquet(
        articles_path=Path("data/processed/wikipedia_articles.parquet"),
        output_path=Path("data/processed/chunks.parquet"),
        chunker=chunker,
    )
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from workbench.observability.tracing import add_span_attributes, start_span

# ---------------------------------------------------------------------------
# Chunk data structure (plain dataclass — lighter than Pydantic for bulk work)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkRecord:
    """A single chunk produced by a chunker."""

    chunk_id: str
    document_id: str
    title: str
    section: str  # empty string if unknown
    text: str
    start_char: int
    end_char: int
    chunker_name: str
    chunker_settings: str  # JSON-ish summary for reproducibility


# ---------------------------------------------------------------------------
# Stable chunk ID
# ---------------------------------------------------------------------------


def make_chunk_id(
    document_id: str,
    start_char: int,
    end_char: int,
    chunker_name: str,
    chunker_settings: str,
) -> str:
    """
    Create a deterministic chunk ID from its defining properties.

    Uses SHA-256 truncated to 16 hex chars.  Two runs with the same
    inputs always produce the same ID.

    Args:
        document_id: Parent document identifier.
        start_char: Start character offset in the document.
        end_char: End character offset in the document.
        chunker_name: Name of the chunker that produced this chunk.
        chunker_settings: String summary of chunker settings.

    Returns:
        Hex string chunk ID (16 characters).
    """
    payload = f"{document_id}|{start_char}|{end_char}|{chunker_name}|{chunker_settings}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    """Interface that all chunkers must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short machine-readable name (e.g. 'fixed_size')."""

    @property
    @abstractmethod
    def settings_str(self) -> str:
        """Deterministic string summarising the chunker's parameters."""

    @abstractmethod
    def chunk(self, doc_id: str, title: str, text: str) -> list[ChunkRecord]:
        """
        Split *text* into chunks.

        Args:
            doc_id: Document identifier.
            title: Document title (stored on every chunk for convenience).
            text: Full document text.

        Returns:
            List of ChunkRecord instances (may be empty for very short docs).
        """


# ---------------------------------------------------------------------------
# Fixed-size chunker
# ---------------------------------------------------------------------------


class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed-size character windows with optional overlap.

    Simple and predictable.  Good baseline for comparison.

    Args:
        chunk_size: Target chunk size in characters.
        overlap: Number of overlapping characters between consecutive chunks.
        min_chunk_size: Discard trailing chunks shorter than this.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    @property
    def name(self) -> str:
        return "fixed_size"

    @property
    def settings_str(self) -> str:
        return (
            f"size={self.chunk_size},overlap={self.overlap},min={self.min_chunk_size}"
        )

    def chunk(self, doc_id: str, title: str, text: str) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        step = self.chunk_size - self.overlap
        pos = 0

        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))
            chunk_text = text[pos:end]

            if len(chunk_text) < self.min_chunk_size and chunks:
                # Trailing runt — skip it
                break

            chunks.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(
                        doc_id, pos, end, self.name, self.settings_str
                    ),
                    document_id=doc_id,
                    title=title,
                    section="",
                    text=chunk_text,
                    start_char=pos,
                    end_char=end,
                    chunker_name=self.name,
                    chunker_settings=self.settings_str,
                )
            )

            pos += step

        return chunks


# ---------------------------------------------------------------------------
# Paragraph-aware chunker
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """
    Split text on blank lines, returning (paragraph_text, start, end).

    Consecutive blank lines are collapsed.  Leading/trailing whitespace
    inside each paragraph is preserved so character offsets stay accurate.
    """
    paragraphs: list[tuple[str, int, int]] = []
    start = 0
    for part in text.split("\n\n"):
        # Account for the "\n\n" separator (2 chars) between parts
        end = start + len(part)
        stripped = part.strip()
        if stripped:
            paragraphs.append((part, start, end))
        start = end + 2  # +2 for the "\n\n" we split on

    return paragraphs


class ParagraphAwareChunker(BaseChunker):
    """
    Split on paragraph boundaries, merging small paragraphs up to a target.

    Produces chunks that read naturally because they never break mid-sentence
    (assuming paragraphs end at sentence boundaries, which is typical for
    Wikipedia).

    Args:
        target_size: Ideal chunk size in characters.
        max_size: Hard upper limit — a single paragraph longer than this
                  is kept as-is (never split mid-paragraph).
        min_chunk_size: Discard chunks shorter than this.
    """

    def __init__(
        self,
        target_size: int = 1000,
        max_size: int = 1500,
        min_chunk_size: int = 100,
    ):
        self.target_size = target_size
        self.max_size = max_size
        self.min_chunk_size = min_chunk_size

    @property
    def name(self) -> str:
        return "paragraph_aware"

    @property
    def settings_str(self) -> str:
        return (
            f"target={self.target_size},max={self.max_size},min={self.min_chunk_size}"
        )

    def chunk(self, doc_id: str, title: str, text: str) -> list[ChunkRecord]:
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            return []

        chunks: list[ChunkRecord] = []

        # Accumulate paragraphs into a buffer
        buf_parts: list[str] = []
        buf_start: int = paragraphs[0][1]
        buf_end: int = paragraphs[0][2]
        buf_len: int = 0

        def _flush() -> None:
            """Emit the current buffer as a chunk."""
            nonlocal buf_parts, buf_start, buf_end, buf_len
            if not buf_parts:
                return
            merged = "\n\n".join(buf_parts)
            if len(merged) >= self.min_chunk_size:
                chunks.append(
                    ChunkRecord(
                        chunk_id=make_chunk_id(
                            doc_id, buf_start, buf_end, self.name, self.settings_str
                        ),
                        document_id=doc_id,
                        title=title,
                        section="",
                        text=merged,
                        start_char=buf_start,
                        end_char=buf_end,
                        chunker_name=self.name,
                        chunker_settings=self.settings_str,
                    )
                )
            buf_parts = []
            buf_len = 0

        for para_text, p_start, p_end in paragraphs:
            para_stripped = para_text.strip()
            para_len = len(para_stripped)

            # Would adding this paragraph exceed the target?
            if buf_parts and buf_len + para_len > self.target_size:
                _flush()
                buf_start = p_start

            buf_parts.append(para_stripped)
            buf_end = p_end
            buf_len += para_len

        # Final flush
        _flush()

        return chunks


# ---------------------------------------------------------------------------
# Chunker factory (for config-driven usage)
# ---------------------------------------------------------------------------

CHUNKERS: dict[str, type[BaseChunker]] = {
    "fixed_size": FixedSizeChunker,
    "paragraph_aware": ParagraphAwareChunker,
}


def create_chunker(name: str, **kwargs) -> BaseChunker:
    """
    Instantiate a chunker by name.

    Args:
        name: Chunker name ('fixed_size' or 'paragraph_aware').
        **kwargs: Passed to the chunker constructor.

    Returns:
        A BaseChunker instance.

    Raises:
        KeyError: If *name* is not a registered chunker.
    """
    if name not in CHUNKERS:
        raise KeyError(f"Unknown chunker: {name!r}.  Available: {list(CHUNKERS)}")
    return CHUNKERS[name](**kwargs)


# ---------------------------------------------------------------------------
# Arrow schema for chunks.parquet
# ---------------------------------------------------------------------------

CHUNKS_SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string()),
        pa.field("document_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("section", pa.string()),
        pa.field("text", pa.string()),
        pa.field("start_char", pa.int64()),
        pa.field("end_char", pa.int64()),
        pa.field("text_length", pa.int64()),
        pa.field("chunker_name", pa.string()),
        pa.field("chunker_settings", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# Build chunks.parquet from wikipedia_articles.parquet
# ---------------------------------------------------------------------------


def build_chunks_parquet(
    articles_path: Path,
    output_path: Path,
    chunker: BaseChunker,
) -> dict:
    """
    Read articles parquet, apply chunker, write chunks parquet.

    Args:
        articles_path: Path to wikipedia_articles.parquet.
        output_path: Where to write chunks.parquet.
        chunker: A BaseChunker instance.

    Returns:
        Summary dict with keys: num_articles, num_chunks,
        avg_chunk_len, chunker_name, duration_s.
    """
    with start_span(
        "data.chunking",
        attributes={
            "articles_path": str(articles_path),
            "output_path": str(output_path),
            "chunker_name": chunker.name,
            "chunker_settings": chunker.settings_str,
        },
    ):
        t0 = time.time()

        articles = pq.read_table(articles_path)
        num_articles = articles.num_rows

        # Columnar accumulators
        col_chunk_id: list[str] = []
        col_doc_id: list[str] = []
        col_title: list[str] = []
        col_section: list[str] = []
        col_text: list[str] = []
        col_start: list[int] = []
        col_end: list[int] = []
        col_text_len: list[int] = []
        col_chunker: list[str] = []
        col_settings: list[str] = []

        page_ids = articles.column("page_id").to_pylist()
        titles = articles.column("title").to_pylist()
        texts = articles.column("text").to_pylist()

        for i, (doc_id, title, text) in enumerate(zip(page_ids, titles, texts)):
            for c in chunker.chunk(doc_id, title, text):
                col_chunk_id.append(c.chunk_id)
                col_doc_id.append(c.document_id)
                col_title.append(c.title)
                col_section.append(c.section)
                col_text.append(c.text)
                col_start.append(c.start_char)
                col_end.append(c.end_char)
                col_text_len.append(len(c.text))
                col_chunker.append(c.chunker_name)
                col_settings.append(c.chunker_settings)

            if (i + 1) % 50_000 == 0:
                print(
                    f"  chunked {i + 1:,}/{num_articles:,} articles, "
                    f"{len(col_chunk_id):,} chunks so far..."
                )

        table = pa.table(
            {
                "chunk_id": pa.array(col_chunk_id, type=pa.string()),
                "document_id": pa.array(col_doc_id, type=pa.string()),
                "title": pa.array(col_title, type=pa.string()),
                "section": pa.array(col_section, type=pa.string()),
                "text": pa.array(col_text, type=pa.string()),
                "start_char": pa.array(col_start, type=pa.int64()),
                "end_char": pa.array(col_end, type=pa.int64()),
                "text_length": pa.array(col_text_len, type=pa.int64()),
                "chunker_name": pa.array(col_chunker, type=pa.string()),
                "chunker_settings": pa.array(col_settings, type=pa.string()),
            },
            schema=CHUNKS_SCHEMA,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)

        duration_s = round(time.time() - t0, 2)
        num_chunks = len(col_chunk_id)
        avg_chunk_len = round(sum(col_text_len) / max(num_chunks, 1), 1)

        add_span_attributes(
            {
                "num_articles": num_articles,
                "num_chunks": num_chunks,
                "avg_chunk_len": avg_chunk_len,
                "duration_s": duration_s,
            }
        )

        summary = {
            "num_articles": num_articles,
            "num_chunks": num_chunks,
            "avg_chunk_len": avg_chunk_len,
            "chunker_name": chunker.name,
            "chunker_settings": chunker.settings_str,
            "output_path": str(output_path),
            "duration_s": duration_s,
        }
        return summary
