"""
Build a full-text search (BM25) index on the LanceDB chunks table.

LanceDB uses Tantivy under the hood for full-text search.  This module
provides a single entry-point that creates (or replaces) the FTS index
on the ``text`` column of the chunks table.

Usage:
    from workbench.data_build.index_keyword import build_keyword_index

    summary = build_keyword_index(
        db_path=Path("data/indexes/active/lancedb"),
    )
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import lancedb

from workbench.observability.tracing import add_span_attributes, start_span

TABLE_NAME = "chunks"


def build_keyword_index(
    db_path: Path,
    table_name: str = TABLE_NAME,
    replace: bool = True,
) -> dict[str, Any]:
    """
    Create a full-text search index on the ``text`` column.

    Args:
        db_path: Path to the LanceDB database directory.
        table_name: Name of the table to index.
        replace: If True, drop and recreate the index if it exists.

    Returns:
        Summary dict with keys: table_name, num_rows, duration_s.
    """
    with start_span(
        "index.keyword.build",
        attributes={
            "db_path": str(db_path),
            "table_name": table_name,
            "replace": replace,
        },
    ):
        t0 = time.time()

        db = lancedb.connect(str(db_path))
        table = db.open_table(table_name)
        num_rows = table.count_rows()

        print(f"Building FTS index on '{table_name}' ({num_rows:,} rows) ...")

        table.create_fts_index("text", replace=replace)

        duration_s = round(time.time() - t0, 2)

        add_span_attributes(
            {
                "num_rows": num_rows,
                "duration_s": duration_s,
            }
        )

        summary = {
            "table_name": table_name,
            "num_rows": num_rows,
            "duration_s": duration_s,
        }
        print(f"FTS index built in {duration_s}s")
        return summary
