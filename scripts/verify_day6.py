"""
Day 6 Setup Verification Script

Run this to verify your Day 6 setup is complete and working.

Usage:
    python scripts/verify_day6.py

This checks:
    1. Streamlit is importable and has chat primitives
    2. UI module exists and is importable
    3. LangGraph .stream() works on a toy graph
    4. Graph builders are importable
    5. Config loads
    6. Components can be built (if LanceDB + model server available)
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def check(label: str, func):
    """Run a check and print result."""
    try:
        func()
        print(f"  ✅ {label}")
        return True
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return False


def main():
    print("=" * 60)
    print("  Day 6 — Streamlit Chat UI Verification")
    print("=" * 60)
    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    print("\n1. Core imports")
    # ------------------------------------------------------------------

    def check_streamlit():
        import streamlit as st

        assert hasattr(st, "chat_message"), "Missing st.chat_message"
        assert hasattr(st, "chat_input"), "Missing st.chat_input"
        assert hasattr(st, "session_state"), "Missing st.session_state"
        assert hasattr(st, "set_page_config"), "Missing st.set_page_config"

    if check("Streamlit installed with chat primitives", check_streamlit):
        passed += 1
    else:
        failed += 1

    def check_langgraph_stream():
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        class S(TypedDict, total=False):
            x: str

        def f(state):
            return {"x": "ok"}

        builder = StateGraph(S)
        builder.add_node("a", f)
        builder.set_entry_point("a")
        builder.add_edge("a", END)
        g = builder.compile()
        events = list(g.stream({"x": ""}))
        assert len(events) >= 1, "stream() should yield events"
        assert "a" in events[0], f"Expected node 'a' in event, got {events[0]}"

    if check("LangGraph .stream() works", check_langgraph_stream):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    print("\n2. UI module")
    # ------------------------------------------------------------------

    def check_ui_package():
        import workbench.ui

        assert workbench.ui is not None

    if check("workbench.ui package importable", check_ui_package):
        passed += 1
    else:
        failed += 1

    def check_app_file_exists():
        app_path = project_root / "src" / "workbench" / "ui" / "streamlit_app.py"
        assert app_path.exists(), f"File not found: {app_path}"

    if check("streamlit_app.py exists", check_app_file_exists):
        passed += 1
    else:
        failed += 1

    def check_run_ui_script():
        script = project_root / "scripts" / "run_ui.sh"
        assert script.exists(), f"File not found: {script}"

    if check("run_ui.sh exists", check_run_ui_script):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    print("\n3. Graph builders")
    # ------------------------------------------------------------------

    def check_rag_graph_import():
        from workbench.graphs.rag_graph import build_rag_graph

        assert callable(build_rag_graph)

    if check("build_rag_graph importable", check_rag_graph_import):
        passed += 1
    else:
        failed += 1

    def check_researcher_graph_import():
        from workbench.graphs.researcher_graph import build_researcher_graph

        assert callable(build_researcher_graph)

    if check("build_researcher_graph importable", check_researcher_graph_import):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    print("\n4. Configuration")
    # ------------------------------------------------------------------

    def check_config():
        from workbench.core.config import load_config

        cfg = load_config()
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "retrieval")
        assert hasattr(cfg, "web_search")
        assert hasattr(cfg, "observability")

    if check("Config loads from defaults.yaml", check_config):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    print("\n5. Component smoke tests (require LanceDB + model server)")
    # ------------------------------------------------------------------

    def check_model_client():
        from workbench.core.config import load_config
        from workbench.models.llamacpp_server import LlamaCppServerModel

        cfg = load_config()
        model = LlamaCppServerModel(
            base_url=cfg.model["base_url"],
            model_name=cfg.model.get("model_name", "local-model"),
        )
        # Just check we can create it — don't call generate()
        assert model is not None

    if check("Model client creates", check_model_client):
        passed += 1
    else:
        failed += 1

    def check_model_reachable():
        import httpx

        from workbench.core.config import load_config

        cfg = load_config()
        url = cfg.model["base_url"]
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            assert r.status_code == 200, f"Server returned {r.status_code}"
        except httpx.ConnectError:
            raise ConnectionError(
                f"Model server not reachable at {url}. "
                "Start it with: bash scripts/start_model_server.sh"
            )

    if check("Model server reachable", check_model_reachable):
        passed += 1
    else:
        failed += 1

    def check_lancedb_exists():
        from workbench.core.config import load_config

        cfg = load_config()
        db_path = Path(cfg.paths["active_index"]) / "lancedb"
        assert db_path.exists(), f"LanceDB not found at {db_path}"

    if check("LanceDB index exists", check_lancedb_exists):
        passed += 1
    else:
        failed += 1

    def check_chunk_store():
        from workbench.core.config import load_config
        from workbench.stores.chunk_store import ChunkStore

        cfg = load_config()
        db_path = Path(cfg.paths["active_index"]) / "lancedb"
        store = ChunkStore(db_path=db_path)
        assert store.count() > 0, "ChunkStore is empty"

    if check("ChunkStore loads with data", check_chunk_store):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    print("\n6. Full graph build (requires all components)")
    # ------------------------------------------------------------------

    def check_rag_graph_builds():
        from workbench.core.config import load_config
        from workbench.data_build.embeddings import SentenceTransformerEmbedder
        from workbench.graphs.rag_graph import build_rag_graph
        from workbench.models.llamacpp_server import LlamaCppServerModel
        from workbench.prompting.prompt_builder import PromptBuilder
        from workbench.retrieval.hybrid import HybridRetriever
        from workbench.retrieval.keyword_search import LanceDBKeywordSearcher
        from workbench.retrieval.vector_search import LanceDBVectorSearcher
        from workbench.stores.chunk_store import ChunkStore
        from workbench.tools.open_chunk import OpenChunkTool
        from workbench.tools.search_wikipedia import SearchWikipediaTool

        cfg = load_config()
        db_path = Path(cfg.paths["active_index"]) / "lancedb"

        model = LlamaCppServerModel(base_url=cfg.model["base_url"])
        embedder = SentenceTransformerEmbedder(model_name=cfg.embeddings["model_name"])
        chunk_store = ChunkStore(db_path=db_path)
        retriever = HybridRetriever(
            keyword_searcher=LanceDBKeywordSearcher(db_path=db_path),
            vector_searcher=LanceDBVectorSearcher(db_path=db_path, embedder=embedder),
            chunk_store=chunk_store,
        )

        graph = build_rag_graph(
            search_tool=SearchWikipediaTool(retriever=retriever),
            open_chunk_tool=OpenChunkTool(chunk_store=chunk_store),
            model=model,
            prompt_builder=PromptBuilder(),
        )
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    if check("RAG graph builds end-to-end", check_rag_graph_builds):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed")

    if failed == 0:
        print("  🎉 All checks passed! Ready to run:")
        print("     streamlit run src/workbench/ui/streamlit_app.py")
        print("     — or —")
        print("     bash scripts/run_ui.sh")
    else:
        print(f"  ⚠️  {failed} check(s) failed.")
        print("  Some failures (model server, LanceDB) are expected")
        print("  if infrastructure isn't running yet.")

    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
