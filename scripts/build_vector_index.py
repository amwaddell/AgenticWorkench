#!/usr/bin/env python3
"""
Build the LanceDB vector index from chunks.parquet.

Usage:
    python scripts/build_vector_index.py

    # With custom paths:
    python scripts/build_vector_index.py \
        --chunks data/processed/chunks.parquet \
        --db-path data/indexes/active/lancedb \
        --batch-size 64

    # Force local embedder (even if config says http):
    python scripts/build_vector_index.py --provider local

This will:
    1. Load chunks from chunks.parquet
    2. Embed all chunk texts using the configured model
    3. Write vectors + metadata to a LanceDB table
    4. Print a summary

Prerequisite:
    pip install lancedb sentence-transformers pyarrow
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path so we can import workbench
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from workbench.core.config import load_config
from workbench.data_build.index_vector import build_vector_index
from workbench.observability.tracing import setup_tracing


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LanceDB vector index")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/chunks.parquet"),
        help="Path to chunks.parquet (default: data/processed/chunks.parquet)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="LanceDB database directory (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding + write batch size (default: from config)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Embedding model name (default: from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for local embedding model: cpu or mps (default: from config)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["http", "local"],
        help="Override embedding provider (default: from config)",
    )
    args = parser.parse_args()

    # --- Load config ---
    config = load_config()
    emb_cfg = config.embeddings

    model_name = args.model_name or emb_cfg.get(
        "model_name", "intfloat/multilingual-e5-base"
    )
    device = args.device or emb_cfg.get("device", "mps")
    batch_size = args.batch_size or emb_cfg.get("batch_size", 64)
    db_path = (
        args.db_path
        or Path(config.paths.get("active_index", "data/indexes/active")) / "lancedb"
    )

    # Determine provider: CLI override > config
    provider = args.provider or emb_cfg.get("provider", "http")

    # --- Setup tracing (sends to Phoenix if running) ---
    setup_tracing(service_name="build-vector-index")

    # --- Validate inputs ---
    if not args.chunks.exists():
        print(f"ERROR: chunks file not found: {args.chunks}")
        print("Run the chunking step first (scripts/build_chunks.py)")
        sys.exit(1)

    # --- Load embedding client/model ---
    if provider == "http":
        from workbench.data_build.http_embedder import HTTPEmbedder

        base_url = emb_cfg.get("base_url", "http://127.0.0.1:8081")
        print(f"Connecting to embedding server: {base_url} ({model_name})")
        t0 = time.time()
        embedder = HTTPEmbedder(
            base_url=base_url,
            model_name=model_name,
            timeout_seconds=emb_cfg.get("timeout_seconds", 120),
        )
        if not embedder.health_check():
            print(f"ERROR: Embedding server not reachable at {base_url}")
            print("Start it with: bash scripts/start_embedding_server.sh")
            print("Or use --provider local to embed in-process.")
            sys.exit(1)
        print(f"Connected in {time.time() - t0:.1f}s")
    else:
        from workbench.data_build.embeddings import SentenceTransformerEmbedder

        print(f"Loading embedding model locally: {model_name} (device={device})")
        t0 = time.time()
        embedder = SentenceTransformerEmbedder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
        print(f"Model loaded in {time.time() - t0:.1f}s  (dim={embedder.dimension})")

    # --- Build index ---
    print("\nBuilding vector index...")
    print(f"  chunks:     {args.chunks}")
    print(f"  db_path:    {db_path}")
    print(f"  provider:   {provider}")
    print(f"  batch_size: {batch_size}")
    print()

    summary = build_vector_index(
        chunks_path=args.chunks,
        db_path=db_path,
        embedder=embedder,
        embed_batch_size=512,
        write_batch_size=512,
    )

    # --- Print summary ---
    print("\n--- Build Summary ---")
    print(json.dumps(summary, indent=2))
    print("Done!")


if __name__ == "__main__":
    main()
