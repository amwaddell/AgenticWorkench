#!/usr/bin/env python3
"""
Preflight check and Phoenix launcher.

Verifies everything is ready before you run the researcher eval:
    1. LLM server is reachable
    2. Vector database exists and has data
    3. Embedding model downloads are cached (triggers download if needed)
    4. Reranker model downloads are cached
    5. Phoenix trace viewer is launched

NOTE: The embedding model and reranker run in-process, so they must be
loaded by each script that uses them. However, running this script first
ensures that model weights are downloaded and cached by HuggingFace,
so subsequent loads in the eval script will be fast (~1-2s from disk
instead of a network download).

Usage:
    python scripts/preflight.py                # check all + start Phoenix
    python scripts/preflight.py --no-phoenix   # check only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from workbench.core.config import load_config


def check_llm_server(cfg) -> bool:
    """Verify the LLM server is reachable."""
    print("[1/4] LLM server...", end=" ", flush=True)
    t0 = time.time()

    from workbench.models.llamacpp_server import LlamaCppServerModel

    model = LlamaCppServerModel(
        base_url=cfg.model["base_url"],
        model_name=cfg.model.get("model_name", "local-model"),
        timeout_seconds=10,
        default_temperature=0.1,
        default_max_tokens=16,
    )

    try:
        resp = model.generate(
            [{"role": "user", "content": "Say OK."}],
            max_tokens=10,
        )
        print(f"OK ({time.time() - t0:.1f}s) — got: {resp.text[:30]!r}")
        return True
    except Exception as e:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        print(f"       {e}")
        print("       → Start your model server: bash scripts/start_model_server.sh")
        return False


def check_vector_db(cfg) -> bool:
    """Verify LanceDB exists and has chunks."""
    print("[2/4] Vector database...", end=" ", flush=True)
    t0 = time.time()

    import lancedb

    db_path = Path(cfg.paths["active_index"]) / "lancedb"
    if not db_path.exists():
        print(f"MISSING — {db_path} does not exist")
        print("       → Run: python scripts/build_vector_index.py")
        return False

    db = lancedb.connect(str(db_path))
    tables = db.table_names()

    if "chunks" not in tables:
        print(f"NO CHUNKS TABLE — tables: {tables}")
        print("       → Run: python scripts/build_vector_index.py")
        return False

    tbl = db.open_table("chunks")
    count = tbl.count_rows()
    print(f"OK ({time.time() - t0:.1f}s) — {count:,} chunks in {len(tables)} tables")
    return True


def check_embedder(cfg) -> bool:
    """Load embedding model (ensures weights are cached)."""
    print("[3/4] Embedding model...", end=" ", flush=True)
    t0 = time.time()

    from workbench.data_build.embeddings import SentenceTransformerEmbedder

    try:
        embedder = SentenceTransformerEmbedder(
            model_name=cfg.embeddings["model_name"],
            device=cfg.embeddings.get("device", "mps"),
            batch_size=cfg.embeddings.get("batch_size", 32),
        )
        # Force model init by embedding a dummy text
        embedder.embed_texts(["preflight check"])
        print(f"OK ({time.time() - t0:.1f}s) — {cfg.embeddings['model_name']}")
        return True
    except Exception as e:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        print(f"       {e}")
        return False


def check_reranker(cfg) -> bool:
    """Load reranker model (ensures weights are cached)."""
    if not cfg.reranking.get("enabled", True):
        print("[4/4] Reranker... DISABLED in config, skipping")
        return True

    print("[4/4] Reranker model...", end=" ", flush=True)
    t0 = time.time()

    from workbench.retrieval.rerankers import CrossEncoderReranker

    try:
        CrossEncoderReranker(model_name=cfg.reranking["model_name"])
        print(f"OK ({time.time() - t0:.1f}s) — {cfg.reranking['model_name']}")
        return True
    except Exception as e:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        print(f"       {e}")
        return False


def start_phoenix() -> subprocess.Popen | None:
    """Start Phoenix in background."""
    print("\nStarting Phoenix on http://localhost:6006 ...", end=" ", flush=True)
    try:
        proc = subprocess.Popen(
            ["phoenix", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        if proc.poll() is None:
            print(f"running (PID {proc.pid})")
            return proc
        else:
            print("already running or exited")
            return None
    except FileNotFoundError:
        print("not found — install with: pip install arize-phoenix arize-phoenix-otel")
        return None


def main():
    parser = argparse.ArgumentParser(description="Preflight check + Phoenix")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--no-phoenix", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path=config_path)

    print("=" * 50)
    print("  Preflight checks")
    print("=" * 50)
    print()

    total_t0 = time.time()
    results = [
        check_llm_server(cfg),
        check_vector_db(cfg),
        check_embedder(cfg),
        check_reranker(cfg),
    ]
    total = time.time() - total_t0

    passed = sum(results)
    total_checks = len(results)

    phoenix_proc = None
    if not args.no_phoenix:
        phoenix_proc = start_phoenix()

    print()
    print("=" * 50)
    print(f"  {passed}/{total_checks} checks passed ({total:.1f}s)")

    if all(results):
        print("  All systems GO")
    else:
        print("  Some checks failed — see above")

    print("=" * 50)
    print()

    if all(results):
        print("Ready. In another terminal run:")
        print("  python scripts/run_researcher_eval.py --question-id hist_001")
        if phoenix_proc:
            print("\nPhoenix: http://localhost:6006")
            print("Press Ctrl+C to stop Phoenix and exit.\n")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down Phoenix...")
                if phoenix_proc.poll() is None:
                    phoenix_proc.terminate()


if __name__ == "__main__":
    main()
