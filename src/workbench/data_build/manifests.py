"""
Manifest reading and writing for data artifacts and indexes.

A manifest is a YAML file that records what was built, when, and from what
configuration — so you can always trace an index or dataset back to its
source and settings.

Example manifest (data/indexes/active/index_manifest.yaml):

    created_at: "2025-02-11T14:30:00+00:00"
    source_parquet: "data/processed/wikipedia_articles.parquet"
    num_articles: 42301
    chunking:
        strategy: heading_aware
        max_chunk_tokens: 512
    indexes:
        keyword: "data/indexes/active/bm25.json"
        vector: "data/indexes/active/faiss.index"
    run_id: "20250211_143000_a1b2c3d4"
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def write_manifest(manifest_path: Path, data: dict[str, Any]) -> Path:
    """
    Write a manifest YAML file.

    Automatically adds ``created_at`` if not already present.

    Args:
        manifest_path: Where to write the manifest.
        data: Manifest content (will be serialized as YAML).

    Returns:
        The path that was written.
    """
    if "created_at" not in data:
        data["created_at"] = datetime.now(UTC).isoformat()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return manifest_path


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    """
    Read a manifest YAML file.

    Args:
        manifest_path: Path to manifest file.

    Returns:
        Parsed manifest dictionary.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty or invalid.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty manifest: {manifest_path}")

    return data


def write_ingest_manifest(
    manifest_path: Path,
    *,
    source_dump: str,
    output_parquet: str,
    num_articles: int,
    total_chars: int,
    skipped: int,
    run_id: str | None = None,
) -> Path:
    """
    Write a manifest specifically for a Wikipedia ingestion run.

    Args:
        manifest_path: Where to write.
        source_dump: Path/name of the source dump.
        output_parquet: Path to the output parquet file.
        num_articles: Number of articles ingested.
        total_chars: Total character count.
        skipped: Number of articles skipped.
        run_id: Optional run ID for traceability.

    Returns:
        Path that was written.
    """
    data = {
        "artifact_type": "wikipedia_articles",
        "source_dump": source_dump,
        "output_parquet": output_parquet,
        "num_articles": num_articles,
        "total_chars": total_chars,
        "skipped_articles": skipped,
    }
    if run_id:
        data["run_id"] = run_id

    return write_manifest(manifest_path, data)
