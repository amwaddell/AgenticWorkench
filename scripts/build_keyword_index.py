#!/usr/bin/env python3
"""
Build the full-text search (BM25) index on the LanceDB chunks table.

Run this AFTER build_vector_index.py has created the LanceDB table.

Usage:
    python scripts/build_keyword_index.py
"""

from pathlib import Path

from workbench.data_build.index_keyword import build_keyword_index
from workbench.observability.tracing import setup_tracing

DB_PATH = Path("data/indexes/active/lancedb")


def main() -> None:
    setup_tracing(service_name="build-keyword-index")

    if not DB_PATH.exists():
        print(f"ERROR: LanceDB not found at {DB_PATH}")
        print("Run scripts/build_vector_index.py first to create the chunks table.")
        raise SystemExit(1)

    summary = build_keyword_index(db_path=DB_PATH)

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
