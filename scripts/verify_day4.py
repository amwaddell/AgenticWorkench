"""
Day 4 Setup Verification Script

Run this to verify your Day 4 setup is complete and working.

Usage:
    python scripts/verify_day4.py
"""

import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def check(label: str, func):
    """Run a check and print result."""
    try:
        func()
        print(f"  ✓ {label}")
        return True
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        return False


def main():
    passed = 0
    failed = 0

    # ---- Import checks ----
    print("\n1. Import checks")

    def _import_base_state():
        from workbench.graphs.base_state import ResearcherState

        assert ResearcherState is not None

    def _import_citations():
        from workbench.prompting.citations import (
            RESEARCHER_STRICT,
        )

        assert RESEARCHER_STRICT.min_citations >= 1

    def _import_researcher_graph():
        from workbench.graphs.researcher_graph import (
            build_researcher_graph,
        )

        assert build_researcher_graph is not None

    def _import_graphs_init():
        pass

    for label, fn in [
        ("ResearcherState imports", _import_base_state),
        ("citations module imports", _import_citations),
        ("researcher_graph imports", _import_researcher_graph),
        ("graphs __init__ exports", _import_graphs_init),
    ]:
        if check(label, fn):
            passed += 1
        else:
            failed += 1

    # ---- Citation validation checks ----
    print("\n2. Citation validation")

    def _extract_citations_works():
        from workbench.prompting.citations import extract_citations

        result = extract_citations(
            "Rome fell [chunk_001] in 27 BC [chunk_002].",
            ["chunk_001", "chunk_002", "chunk_003"],
        )
        assert result == ["chunk_001", "chunk_002"], f"Got {result}"

    def _strict_validation_catches_no_citations():
        from workbench.prompting.citations import (
            RESEARCHER_STRICT,
            validate_citations,
        )

        vr = validate_citations("No citations here.", ["c1"], RESEARCHER_STRICT)
        assert not vr.valid, "Should fail with no citations"

    def _strict_validation_catches_hallucinated():
        from workbench.prompting.citations import (
            RESEARCHER_STRICT,
            validate_citations,
        )

        vr = validate_citations("[fake_id] [c1]", ["c1"], RESEARCHER_STRICT)
        assert not vr.valid, "Should fail with hallucinated IDs"
        assert "fake_id" in vr.hallucinated_ids

    def _rag_default_accepts_no_citations():
        from workbench.prompting.citations import RAG_DEFAULT, validate_citations

        vr = validate_citations("No citations.", ["c1"], RAG_DEFAULT)
        assert vr.valid, "RAG_DEFAULT should accept no citations"

    for label, fn in [
        ("extract_citations works", _extract_citations_works),
        ("STRICT catches zero citations", _strict_validation_catches_no_citations),
        ("STRICT catches hallucinated IDs", _strict_validation_catches_hallucinated),
        ("RAG_DEFAULT is lenient", _rag_default_accepts_no_citations),
    ]:
        if check(label, fn):
            passed += 1
        else:
            failed += 1

    # ---- Graph compilation checks ----
    print("\n3. Graph compilation")

    def _build_researcher_graph():
        from workbench.core.types import ModelResponse
        from workbench.graphs.researcher_graph import build_researcher_graph

        class DummySearch:
            def execute(self, **kw):
                return {"chunks": [], "count": 0}

        class DummyOpen:
            def execute(self, **kw):
                return {"found": False, "chunk": None, "text_length": 0}

        class DummyTimeline:
            def execute(self, **kw):
                return {
                    "timeline": [],
                    "count": 0,
                    "cited_chunk_ids": [],
                    "raw_model_output": "",
                    "prompt_version": "test",
                }

        class DummyModel:
            def generate(self, messages, **kw):
                return ModelResponse(
                    text="test", tokens_in=1, tokens_out=1, latency_ms=1.0
                )

        g = build_researcher_graph(
            search_tool=DummySearch(),
            open_chunk_tool=DummyOpen(),
            timeline_tool=DummyTimeline(),
            model=DummyModel(),
        )
        assert hasattr(g, "invoke"), "Graph should have invoke()"

    def _invoke_researcher_graph():
        from workbench.core.types import ModelResponse
        from workbench.graphs.researcher_graph import build_researcher_graph

        class DummySearch:
            def execute(self, **kw):
                return {"chunks": [], "count": 0}

        class DummyOpen:
            def execute(self, **kw):
                return {"found": False, "chunk": None, "text_length": 0}

        class DummyTimeline:
            def execute(self, **kw):
                return {
                    "timeline": [],
                    "count": 0,
                    "cited_chunk_ids": [],
                    "raw_model_output": "",
                    "prompt_version": "test",
                }

        class DummyModel:
            def generate(self, messages, **kw):
                return ModelResponse(
                    text="I don't have enough info.",
                    tokens_in=5,
                    tokens_out=5,
                    latency_ms=10.0,
                )

        g = build_researcher_graph(
            search_tool=DummySearch(),
            open_chunk_tool=DummyOpen(),
            timeline_tool=DummyTimeline(),
            model=DummyModel(),
        )
        result = g.invoke({"question": "test"})
        assert "answer_text" in result
        assert "citation_validation" in result

    for label, fn in [
        ("build_researcher_graph compiles", _build_researcher_graph),
        ("researcher graph invoke works", _invoke_researcher_graph),
    ]:
        if check(label, fn):
            passed += 1
        else:
            failed += 1

    # ---- Backward compat checks ----
    print("\n4. Backward compatibility")

    def _rag_graph_still_works():
        from workbench.graphs.rag_graph import build_rag_graph, extract_citations

        assert extract_citations is not None
        assert build_rag_graph is not None

    def _rag_state_unchanged():
        from workbench.graphs.base_state import RAGState

        # RAGState should still have the expected fields
        hints = RAGState.__annotations__
        assert "question" in hints
        assert "answer_text" in hints
        assert "citations" in hints

    for label, fn in [
        ("rag_graph imports unchanged", _rag_graph_still_works),
        ("RAGState fields intact", _rag_state_unchanged),
    ]:
        if check(label, fn):
            passed += 1
        else:
            failed += 1

    # ---- Summary ----
    total = passed + failed
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed")
    if failed:
        print(f"  {failed} FAILED — see errors above")
        sys.exit(1)
    else:
        print("  All checks passed! Day 4 is ready.")
        sys.exit(0)


if __name__ == "__main__":
    main()
