#!/usr/bin/env python3
"""
Run Wikipedia ingestion pipeline (direct XML dump parsing).

Usage:
    python scripts/run_ingest_wikipedia.py

Prerequisites:
    1. Download:   bash scripts/download_wikipedia.sh
    2. (Optional)  phoenix serve   — to view traces in the Phoenix UI

This script will:
    - Parse the .xml.bz2 dump directly (no WikiExtractor needed)
    - Strip wiki markup to produce clean plain text
    - Write data/processed/wikipedia_articles.parquet
    - Write an ingest manifest alongside it
    - Log events and send trace spans
"""

import sys
from pathlib import Path

# Add src to path so we can import workbench
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from workbench.core.config import create_config_snapshot, load_config
from workbench.core.run_context import run_context
from workbench.data_build.ingest_wikipedia import ingest_wikipedia_dump
from workbench.data_build.manifests import write_ingest_manifest
from workbench.observability.logging import get_logger
from workbench.observability.tracing import setup_tracing, shutdown_tracing

# Try to set up Phoenix exporter; fall back to no export
try:
    from workbench.observability.phoenix import create_phoenix_exporter

    exporter = create_phoenix_exporter()
except Exception:
    exporter = None
    print("(Phoenix exporter not available — tracing to console only)")


def main():
    # ── Load config ────────────────────────────────────────────────────
    config = load_config()
    snapshot = create_config_snapshot(config)

    # ── Set up tracing ─────────────────────────────────────────────────
    setup_tracing(
        service_name="agentic-workbench",
        exporter=exporter,
        console_export=False,
    )

    # ── Paths ──────────────────────────────────────────────────────────
    dump_path = (
        Path(config.paths.get("raw_data", "data/raw"))
        / "wikipedia"
        / "simplewiki-latest-pages-articles.xml.bz2"
    )
    output_path = (
        Path(config.paths.get("processed_data", "data/processed"))
        / "wikipedia_articles.parquet"
    )
    manifest_path = output_path.parent / "wikipedia_articles_manifest.yaml"

    if not dump_path.exists():
        print(f"ERROR: Dump file not found: {dump_path}")
        print("Run first:  bash scripts/download_wikipedia.sh")
        sys.exit(1)

    # ── Run ingestion inside a traced context ──────────────────────────
    with run_context(config_snapshot=snapshot, run_type="ingest") as ctx:
        logger = get_logger(ctx.run_id)
        logger.log_run_started(ctx.run_id, ctx.run_type, snapshot)
        logger.log_component_started("ingest_wikipedia", "data_build")

        print(f"Run ID:     {ctx.run_id}")
        print(f"Dump file:  {dump_path}")
        print(f"Output:     {output_path}")
        print()

        summary = ingest_wikipedia_dump(
            dump_path=dump_path,
            output_path=output_path,
            source_label="simplewiki-latest",
        )

        # Write manifest
        write_ingest_manifest(
            manifest_path,
            source_dump=str(dump_path),
            output_parquet=str(output_path),
            num_articles=summary["num_articles"],
            total_chars=summary["total_chars"],
            skipped=summary["skipped"],
            run_id=ctx.run_id,
        )

        logger.log_component_finished(
            "ingest_wikipedia",
            "data_build",
            duration_ms=summary["duration_s"] * 1000,
            result_summary=summary,
        )

        print()
        print("=== Ingestion Complete ===")
        print(f"  Articles:      {summary['num_articles']:,}")
        print(f"  Total chars:   {summary['total_chars']:,}")
        print(f"  Skipped:       {summary['skipped']:,}")
        print(f"  Duration:      {summary['duration_s']:.1f}s")
        print(f"  Parquet:       {summary['output_path']}")
        print(f"  Manifest:      {manifest_path}")
        print(f"  Log:           runs/logs/{ctx.run_id}.jsonl")

    shutdown_tracing()


if __name__ == "__main__":
    main()
