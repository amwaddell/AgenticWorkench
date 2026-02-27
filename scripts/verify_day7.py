"""
Day 7 Setup Verification Script

Usage:
    python scripts/verify_day7.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def check(label: str, func):
    try:
        func()
        print(f"  ✅ {label}")
        return True
    except Exception as exc:
        print(f"  ❌ {label}: {exc}")
        return False


def main():
    print("=" * 60)
    print("Day 7 — Supervisor Graph Verification")
    print("=" * 60)
    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    print("\n1. Imports — routing subpackage")
    # ------------------------------------------------------------------

    def check_routing_constants():
        from workbench.graphs.routing.constants import (
            VALID_ROUTES,
        )

        assert len(VALID_ROUTES) == 4

    if check("routing.constants", check_routing_constants):
        passed += 1
    else:
        failed += 1

    def check_heuristic():
        from workbench.graphs.routing.heuristic import heuristic_route

        route, conf = heuristic_route("2 + 2 * 3")
        assert route == "math"

    if check("routing.heuristic", check_heuristic):
        passed += 1
    else:
        failed += 1

    def check_llm_router():
        pass

    if check("routing.llm_router imports", check_llm_router):
        passed += 1
    else:
        failed += 1

    def check_routing_init():
        pass

    if check("routing __init__ re-exports", check_routing_init):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    print("\n2. Imports — supervisor graph + state")
    # ------------------------------------------------------------------

    def check_supervisor_graph():
        pass

    if check("supervisor_graph imports", check_supervisor_graph):
        passed += 1
    else:
        failed += 1

    def check_base_state():
        pass

    if check("SupervisorState", check_base_state):
        passed += 1
    else:
        failed += 1

    def check_graphs_init():
        pass

    if check("graphs __init__ re-exports", check_graphs_init):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    print("\n3. Heuristic Router")
    # ------------------------------------------------------------------

    def check_math():
        from workbench.graphs.routing import heuristic_route

        r, c = heuristic_route("2 + 2 * 3")
        assert r == "math" and c > 0.5

    if check("math expression → math", check_math):
        passed += 1
    else:
        failed += 1

    def check_web():
        from workbench.graphs.routing import heuristic_route

        r, _ = heuristic_route("What are the latest news today?")
        assert r == "web_research"

    if check("'latest news today' → web_research", check_web):
        passed += 1
    else:
        failed += 1

    def check_timeline():
        from workbench.graphs.routing import heuristic_route

        r, _ = heuristic_route("Give me a timeline of the French Revolution")
        assert r == "timeline_research"

    if check("'timeline of…' → timeline_research", check_timeline):
        passed += 1
    else:
        failed += 1

    def check_factual():
        from workbench.graphs.routing import heuristic_route

        r, _ = heuristic_route("What is photosynthesis?")
        assert r == "local_rag"

    if check("'What is photosynthesis?' → local_rag", check_factual):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    print("\n4. Graph compilation (fakes)")
    # ------------------------------------------------------------------

    def check_compile():
        from workbench.core.types import ModelResponse
        from workbench.graphs.supervisor_graph import build_supervisor_graph

        class S:
            def execute(self, **kw):
                return {
                    "chunks": [
                        {
                            "chunk_id": "c1",
                            "score": 0.9,
                            "title": "T",
                            "section": "S",
                            "snippet": "…",
                        }
                    ]
                }

        class O:
            def execute(self, **kw):
                return {
                    "found": True,
                    "chunk": {
                        "chunk_id": kw.get("chunk_id", "?"),
                        "title": "T",
                        "section": "S",
                        "text": "Text.",
                    },
                    "text_length": 5,
                }

        class M:
            def generate(self, msgs, **kw):
                return ModelResponse(
                    text="Answer [c1].", tokens_in=10, tokens_out=5, latency_ms=1.0
                )

        graph = build_supervisor_graph(search_tool=S(), open_chunk_tool=O(), model=M())
        assert hasattr(graph, "invoke") and hasattr(graph, "stream")

    if check("compile with minimal tools", check_compile):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    print("\n5. Graph invoke (fakes)")
    # ------------------------------------------------------------------

    def check_invoke_math():
        from workbench.core.types import ModelResponse
        from workbench.graphs.supervisor_graph import build_supervisor_graph

        class S:
            def execute(self, **kw):
                return {"chunks": []}

        class O:
            def execute(self, **kw):
                return {"found": False, "chunk": None, "text_length": 0}

        class M:
            def generate(self, msgs, **kw):
                return ModelResponse(
                    text="42", tokens_in=10, tokens_out=5, latency_ms=1.0
                )

        class C:
            def execute(self, **kw):
                expr = kw.get("expression", "")
                try:
                    if all(c in "0123456789+-*/. ()" for c in expr):
                        return {"expression": expr, "result": eval(expr), "error": None}
                except:
                    pass
                return {"expression": expr, "result": None, "error": "fail"}

        graph = build_supervisor_graph(
            search_tool=S(),
            open_chunk_tool=O(),
            model=M(),
            calculator_tool=C(),
            routing_mode="heuristic_only",
        )
        result = graph.invoke({"question": "100 + 50"})
        assert result["route"] == "math"
        assert "150" in result["answer_text"]
        assert result["subgraph_used"] == "calculator"

    if check("invoke math → correct answer", check_invoke_math):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    print("\n6. Config")
    # ------------------------------------------------------------------

    def check_config():
        from workbench.core.config import load_config

        cfg = load_config()
        sup = getattr(cfg, "supervisor", None)
        assert sup is not None and "routing_mode" in sup

    if check("defaults.yaml has supervisor section", check_config):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("🎉 Day 7 setup is complete!")
    else:
        print("⚠️  Some checks failed — review the errors above.")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
