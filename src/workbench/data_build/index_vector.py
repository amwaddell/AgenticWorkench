"""
Build a LanceDB vector index from chunks.parquet.

Reads the chunk dataset, embeds text in batches, and writes to a LanceDB
table.  Designed to be called from scripts/build_vector_index.py but
also usable programmatically.

The resulting LanceDB table has columns:
    chunk_id, document_id, title, section, text, vector

Usage:
    from workbench.data_build.index_vector import build_vector_index

    summary = build_vector_index(
        chunks_path=Path("data/processed/chunks.parquet"),
        db_path=Path("data/indexes/active/lancedb"),
        embedder=my_embedder,
    )
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import lancedb
import pyarrow.parquet as pq
from tqdm import tqdm

from workbench.data_build.embeddings import SentenceTransformerEmbedder
from workbench.observability.tracing import add_span_attributes, start_span

# LanceDB table name
TABLE_NAME = "chunks"


def build_vector_index(
    chunks_path: Path,
    db_path: Path,
    embedder: SentenceTransformerEmbedder,
    embed_batch_size: int = 512,
    write_batch_size: int = 512,
    overwrite: bool = True,
) -> dict[str, Any]:
    """
    Build a LanceDB vector index from a chunks parquet file.

    Args:
        chunks_path: Path to chunks.parquet.
        db_path: Directory for the LanceDB database.
        embedder: Embedder instance to generate vectors.
        embed_batch_size: Number of texts per embedding call (controls memory).
        write_batch_size: Number of rows per LanceDB write batch.
        overwrite: If True, drop existing table before writing.

    Returns:
        Summary dict with keys: num_chunks, embedding_dim,
        db_path, table_name, duration_s.
    """
    with start_span(
        "index.vector.build",
        attributes={
            "chunks_path": str(chunks_path),
            "db_path": str(db_path),
            "model_name": embedder.model_name,
            "embed_batch_size": embed_batch_size,
            "write_batch_size": write_batch_size,
        },
    ):
        t0 = time.time()

        # ------------------------------------------------------------------
        # 1.  Read chunks
        # ------------------------------------------------------------------
        chunks_table = pq.read_table(chunks_path)
        num_chunks = chunks_table.num_rows
        print(f"Read {num_chunks:,} chunks from {chunks_path}")

        chunk_ids = chunks_table.column("chunk_id").to_pylist()
        doc_ids = chunks_table.column("document_id").to_pylist()
        titles = chunks_table.column("title").to_pylist()
        sections = chunks_table.column("section").to_pylist()
        texts = chunks_table.column("text").to_pylist()

        # ------------------------------------------------------------------
        # 2.  Embed in batches (chunked to manage MPS memory)
        # ------------------------------------------------------------------
        print(
            f"Embedding {num_chunks:,} chunks "
            f"(model batch={embedder.batch_size}, "
            f"embed batches of {embed_batch_size})..."
        )

        all_vectors: list[list[float]] = []

        with tqdm(
            total=num_chunks,
            desc="Embedding",
            unit="chunks",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ),
        ) as pbar:
            for batch_start in range(0, num_chunks, embed_batch_size):
                batch_end = min(batch_start + embed_batch_size, num_chunks)
                batch_vecs = embedder.embed_texts(texts[batch_start:batch_end])
                all_vectors.extend(batch_vecs)
                pbar.update(batch_end - batch_start)

        embedding_dim = len(all_vectors[0]) if all_vectors else 0
        embed_time = time.time() - t0
        print(
            f"Embedding complete — {len(all_vectors):,} vectors, "
            f"dim={embedding_dim}, {embed_time:.1f}s"
        )

        # ------------------------------------------------------------------
        # 3.  Open / create LanceDB database
        # ------------------------------------------------------------------
        db_path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(db_path))

        if overwrite and TABLE_NAME in db.table_names():
            db.drop_table(TABLE_NAME)
            print(f"Dropped existing table '{TABLE_NAME}'")

        # ------------------------------------------------------------------
        # 4.  Write in batches
        # ------------------------------------------------------------------
        table_handle = None
        rows_written = 0

        with tqdm(
            total=num_chunks,
            desc="Writing  ",
            unit="rows",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ),
        ) as pbar:
            for start in range(0, num_chunks, write_batch_size):
                end = min(start + write_batch_size, num_chunks)

                with start_span(
                    "index.vector.batch_write",
                    attributes={
                        "batch_start": start,
                        "batch_end": end,
                        "batch_size": end - start,
                    },
                ):
                    batch_data = [
                        {
                            "chunk_id": chunk_ids[i],
                            "document_id": doc_ids[i],
                            "title": titles[i],
                            "section": sections[i],
                            "text": texts[i],
                            "vector": all_vectors[i],
                        }
                        for i in range(start, end)
                    ]

                    if table_handle is None:
                        table_handle = db.create_table(TABLE_NAME, data=batch_data)
                    else:
                        table_handle.add(batch_data)

                    rows_written += len(batch_data)
                    pbar.update(len(batch_data))

        duration_s = round(time.time() - t0, 2)

        add_span_attributes(
            {
                "num_chunks": num_chunks,
                "embedding_dim": embedding_dim,
                "rows_written": rows_written,
                "duration_s": duration_s,
            }
        )

        summary = {
            "num_chunks": num_chunks,
            "embedding_dim": embedding_dim,
            "db_path": str(db_path),
            "table_name": TABLE_NAME,
            "rows_written": rows_written,
            "duration_s": duration_s,
        }
        print(f"\nVector index built: {rows_written:,} rows in {duration_s}s")
        return summary
