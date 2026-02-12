#!/usr/bin/env python3
"""
Build chunks.parquet from wikipedia_articles.parquet.

Usage:
    python scripts/build_chunks.py
    python scripts/build_chunks.py --chunker paragraph_aware --target-size 1000
    python scripts/build_chunks.py --chunker fixed_size --chunk-size 800 --overlap 200

Prerequisites:
    data/processed/wikipedia_articles.parquet must exist (Day 4).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from workbench.core.config import create_config_snapshot, load_config
from workbench.core.run_context import run_context
from workbench.data_build.chunking import (
    FixedSizeChunker,
    ParagraphAwareChunker,
    build_chunks_parquet,
)
from workbench.data_build.manifests import write_manifest
from workbench.observability.logging import get_logger
from workbench.observability.tracing import setup_tracing, shutdown_tracing

# Try Phoenix exporter
try:
    from workbench.observability.phoenix import create_phoenix_exporter

    exporter = create_phoenix_exporter()
except Exception:
    exporter = None
    print("(Phoenix exporter not available — tracing locally only)")


def main():
    parser = argparse.ArgumentParser(description="Build chunks.parquet")
    parser.add_argument(
        "--chunker",
        default="paragraph_aware",
        choices=["fixed_size", "paragraph_aware"],
        help="Chunker strategy (default: paragraph_aware)",
    )
    # Fixed-size options
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Fixed-size: chunk size in chars"
    )
    parser.add_argument(
        "--overlap", type=int, default=200, help="Fixed-size: overlap in chars"
    )
    # Paragraph-aware options
    parser.add_argument(
        "--target-size",
        type=int,
        default=1000,
        help="Paragraph-aware: target size in chars",
    )
    parser.add_argument(
        "--max-size", type=int, default=1500, help="Paragraph-aware: max size in chars"
    )
    # Common
    parser.add_argument(
        "--min-chunk-size", type=int, default=100, help="Minimum chunk size in chars"
    )

    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────────
    config = load_config()
    snapshot = create_config_snapshot(config)

    setup_tracing(service_name="agentic-workbench", exporter=exporter)

    # ── Paths ──────────────────────────────────────────────────────────
    articles_path = (
        Path(config.paths.get("processed_data", "data/processed"))
        / "wikipedia_articles.parquet"
    )
    output_path = (
        Path(config.paths.get("processed_data", "data/processed")) / "chunks.parquet"
    )
    manifest_path = output_path.parent / "chunks_manifest.yaml"

    if not articles_path.exists():
        print(f"ERROR: Articles file not found: {articles_path}")
        print("Run Day 4 ingestion first:  python scripts/run_ingest_wikipedia.py")
        sys.exit(1)

    # ── Create chunker ─────────────────────────────────────────────────
    if args.chunker == "fixed_size":
        chunker = FixedSizeChunker(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            min_chunk_size=args.min_chunk_size,
        )
    else:
        chunker = ParagraphAwareChunker(
            target_size=args.target_size,
            max_size=args.max_size,
            min_chunk_size=args.min_chunk_size,
        )

    # ── Run ────────────────────────────────────────────────────────────
    with run_context(config_snapshot=snapshot, run_type="chunking") as ctx:
        logger = get_logger(ctx.run_id)
        logger.log_run_started(ctx.run_id, ctx.run_type, snapshot)
        logger.log_component_started(
            "build_chunks",
            "data_build",
            {
                "chunker": chunker.name,
                "settings": chunker.settings_str,
            },
        )

        print(f"Run ID:      {ctx.run_id}")
        print(f"Chunker:     {chunker.name} ({chunker.settings_str})")
        print(f"Articles:    {articles_path}")
        print(f"Output:      {output_path}")
        print()

        summary = build_chunks_parquet(
            articles_path=articles_path,
            output_path=output_path,
            chunker=chunker,
        )

        # Write manifest
        write_manifest(
            manifest_path,
            {
                "artifact_type": "chunks",
                "source_articles": str(articles_path),
                "chunker_name": summary["chunker_name"],
                "chunker_settings": summary["chunker_settings"],
                "num_articles": summary["num_articles"],
                "num_chunks": summary["num_chunks"],
                "avg_chunk_len": summary["avg_chunk_len"],
                "run_id": ctx.run_id,
            },
        )

        logger.log_component_finished(
            "build_chunks",
            "data_build",
            duration_ms=summary["duration_s"] * 1000,
            result_summary=summary,
        )

        print()
        print("=== Chunking Complete ===")
        print(f"  Articles:        {summary['num_articles']:,}")
        print(f"  Chunks:          {summary['num_chunks']:,}")
        print(f"  Avg chunk len:   {summary['avg_chunk_len']:.0f} chars")
        print(
            f"  Chunker:         {summary['chunker_name']} ({summary['chunker_settings']})"
        )
        print(f"  Duration:        {summary['duration_s']:.1f}s")
        print(f"  Output:          {summary['output_path']}")
        print(f"  Manifest:        {manifest_path}")

    shutdown_tracing()


if __name__ == "__main__":
    main()
