#!/usr/bin/env python3
"""
Unified preflight check for the Agentic Workbench.

Verifies that all components are ready to run:
    1. LLM server is reachable
    2. Vector database exists and has data
    3. Embedding model loads successfully
    4. Reranker model loads successfully
    5. LangGraph graphs compile (with fakes)
    6. Streamlit is installed
    7. Phoenix trace viewer is launched (optional)

Replaces the per-day ``verify_day*.py`` scripts with a single command.

Usage:
    python scripts/preflight.py                # check all + start Phoenix
    python scripts/preflight.py --no-phoenix   # check only
    python scripts/preflight.py --quick        # skip model downloads
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from workbench.core.config import load_config

# ------------------------------------------------------------------
# Individual checks
# ------------------------------------------------------------------


def check_llm_server(cfg) -> bool:
    """Verify the LLM server is reachable."""
    print("[1/6] LLM server...", end=" ", flush=True)
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
    print("[2/6] Vector database...", end=" ", flush=True)
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
    print("[3/6] Embedding model...", end=" ", flush=True)
    t0 = time.time()

    from workbench.data_build.embeddings import SentenceTransformerEmbedder

    try:
        embedder = SentenceTransformerEmbedder(
            model_name=cfg.embeddings["model_name"],
            device=cfg.embeddings.get("device", "mps"),
            batch_size=cfg.embeddings.get("batch_size", 32),
        )
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
        print("[4/6] Reranker... DISABLED in config, skipping")
        return True

    print("[4/6] Reranker model...", end=" ", flush=True)
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


def check_graphs_compile() -> bool:
    """Verify that all three graphs compile with fake components."""
    print("[5/6] Graph compilation...", end=" ", flush=True)
    t0 = time.time()

    try:
        from workbench.core.types import ModelResponse
        from workbench.graphs.rag_graph import build_rag_graph
        from workbench.graphs.researcher_graph import build_researcher_graph
        from workbench.graphs.supervisor_graph import build_supervisor_graph

        class FakeSearch:
            def execute(self, **kw):
                return {
                    "chunks": [
                        {
                            "chunk_id": "c1",
                            "score": 0.9,
                            "title": "T",
                            "section": "S",
                            "snippet": "...",
                        }
                    ]
                }

        class FakeOpen:
            def execute(self, **kw):
                return {
                    "found": True,
                    "chunk": {
                        "chunk_id": kw.get("chunk_id", "c1"),
                        "title": "T",
                        "section": "S",
                        "text": "Text.",
                    },
                    "text_length": 5,
                }

        class FakeTimeline:
            def execute(self, **kw):
                return {
                    "timeline": [
                        {"date": "100 BC", "event": "E", "supporting_chunk_ids": ["c1"]}
                    ],
                    "raw_model_output": "",
                    "cited_chunk_ids": ["c1"],
                }

        class FakeModel:
            def generate(self, msgs, **kw):
                return ModelResponse(
                    text="Answer [c1].", tokens_in=10, tokens_out=5, latency_ms=1.0
                )

        s, o, m, tl = FakeSearch(), FakeOpen(), FakeModel(), FakeTimeline()

        g1 = build_rag_graph(search_tool=s, open_chunk_tool=o, model=m)
        g2 = build_researcher_graph(
            search_tool=s, open_chunk_tool=o, timeline_tool=tl, model=m
        )
        g3 = build_supervisor_graph(search_tool=s, open_chunk_tool=o, model=m)

        for g in (g1, g2, g3):
            assert hasattr(g, "invoke"), "Graph missing invoke()"
            assert hasattr(g, "stream"), "Graph missing stream()"

        print(
            f"OK ({time.time() - t0:.1f}s) — rag_graph, researcher_graph, supervisor_graph"
        )
        return True

    except Exception as e:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        print(f"       {e}")
        return False


def check_streamlit() -> bool:
    """Verify that Streamlit is importable."""
    print("[6/6] Streamlit...", end=" ", flush=True)
    try:
        import streamlit  # noqa: F401

        print(f"OK — v{streamlit.__version__}")
        return True
    except ImportError:
        print("NOT INSTALLED")
        print("       → pip install streamlit")
        return False


# ------------------------------------------------------------------
# Phoenix launcher
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Preflight check + Phoenix")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--no-phoenix", action="store_true", help="Skip starting Phoenix"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip model-download checks (embedder, reranker)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path=config_path)

    print("=" * 55)
    print("  Agentic Workbench — Preflight Checks")
    print("=" * 55)
    print()

    total_t0 = time.time()

    results = [check_llm_server(cfg), check_vector_db(cfg)]

    if args.quick:
        print("[3/6] Embedding model... SKIPPED (--quick)")
        results.append(True)
        print("[4/6] Reranker... SKIPPED (--quick)")
        results.append(True)
    else:
        results.append(check_embedder(cfg))
        results.append(check_reranker(cfg))

    results.append(check_graphs_compile())
    results.append(check_streamlit())

    total = time.time() - total_t0
    passed = sum(results)
    total_checks = len(results)

    phoenix_proc = None
    if not args.no_phoenix:
        phoenix_proc = start_phoenix()

    print()
    print("=" * 55)
    print(f"  {passed}/{total_checks} checks passed ({total:.1f}s)")

    if all(results):
        print("  All systems GO ✅")
    else:
        print("  Some checks failed — see above ⚠️")

    print("=" * 55)
    print()

    if all(results):
        print("Ready to use. Pick one:")
        print()
        print("  Chat UI:")
        print("    bash scripts/run_ui.sh")
        print()
        print("  Single question (CLI):")
        print('    python scripts/run_agent_chat.py "Your question here"')
        print()
        print("  Evaluation:")
        print("    python scripts/run_researcher_eval.py --question-id hist_001")
        print()
        if phoenix_proc:
            print("Phoenix: http://localhost:6006")
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
