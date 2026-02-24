"""
Day 5 Setup Verification Script

Run this to verify your Day 5 setup is complete and working.

Usage:
    python scripts/verify_day5.py
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
    total = 0

    print("=" * 60)
    print("Day 5 — Verification: Tools + Graph Wiring")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Import checks
    # ----------------------------------------------------------------
    print("\n1. Import checks")

    def check_web_search_import():
        pass

    def check_source_fetch_import():
        pass

    def check_calculator_import():
        pass

    def check_date_normalizer_import():
        pass

    def check_tools_init_import():
        pass

    def check_graphs_init_import():
        pass

    for label, fn in [
        ("WebSearchTool imports", check_web_search_import),
        ("SourceFetchTool imports", check_source_fetch_import),
        ("CalculatorTool imports", check_calculator_import),
        ("DateNormalizerTool imports", check_date_normalizer_import),
        ("tools __init__ re-exports all", check_tools_init_import),
        ("graphs __init__ has extract_web_citations", check_graphs_init_import),
    ]:
        total += 1
        if check(label, fn):
            passed += 1
        else:
            failed += 1

    # ----------------------------------------------------------------
    # 2. Tool construction
    # ----------------------------------------------------------------
    print("\n2. Tool construction")

    def check_web_search_construct():
        from workbench.tools.web_search import WebSearchTool

        t = WebSearchTool(provider="duckduckgo")
        assert t.name == "web_search"
        assert t.provider == "duckduckgo"

    def check_source_fetch_construct():
        from workbench.tools.source_fetch import SourceFetchTool

        t = SourceFetchTool(timeout=10.0)
        assert t.name == "source_fetch"

    def check_calculator_construct():
        from workbench.tools.utility.calculator import CalculatorTool

        t = CalculatorTool()
        assert t.name == "calculator"

    def check_date_normalizer_construct():
        from workbench.tools.utility.dates import DateNormalizerTool

        t = DateNormalizerTool()
        assert t.name == "date_normalizer"

    for label, fn in [
        ("WebSearchTool(provider='duckduckgo')", check_web_search_construct),
        ("SourceFetchTool(timeout=10)", check_source_fetch_construct),
        ("CalculatorTool()", check_calculator_construct),
        ("DateNormalizerTool()", check_date_normalizer_construct),
    ]:
        total += 1
        if check(label, fn):
            passed += 1
        else:
            failed += 1

    # ----------------------------------------------------------------
    # 3. Calculator execution
    # ----------------------------------------------------------------
    print("\n3. Calculator execution")

    def check_calc_basic():
        from workbench.tools.utility.calculator import CalculatorTool

        t = CalculatorTool()
        r = t.execute(expression="2 + 3 * 4")
        assert r["result"] == 14
        assert r["error"] is None

    def check_calc_sqrt():
        from workbench.tools.utility.calculator import CalculatorTool

        t = CalculatorTool()
        r = t.execute(expression="sqrt(144)")
        assert r["result"] == 12.0

    def check_calc_safety():
        from workbench.tools.utility.calculator import CalculatorTool

        t = CalculatorTool()
        r = t.execute(expression="__import__('os')")
        assert r["error"] is not None
        assert r["result"] is None

    for label, fn in [
        ("2 + 3 * 4 = 14", check_calc_basic),
        ("sqrt(144) = 12", check_calc_sqrt),
        ("Rejects __import__", check_calc_safety),
    ]:
        total += 1
        if check(label, fn):
            passed += 1
        else:
            failed += 1

    # ----------------------------------------------------------------
    # 4. Date normalizer execution
    # ----------------------------------------------------------------
    print("\n4. Date normalizer execution")

    def check_date_iso():
        from workbench.tools.utility.dates import DateNormalizerTool

        t = DateNormalizerTool()
        r = t.execute(date_string="2024-03-15")
        assert r["iso"] == "2024-03-15"

    def check_date_bc():
        from workbench.tools.utility.dates import DateNormalizerTool

        t = DateNormalizerTool()
        r = t.execute(date_string="March 15, 44 BC")
        assert r["year"] == -44
        assert r["era"] == "BC"

    def check_date_circa():
        from workbench.tools.utility.dates import DateNormalizerTool

        t = DateNormalizerTool()
        r = t.execute(date_string="circa 500 AD")
        assert r["approximate"] is True
        assert r["year"] == 500

    for label, fn in [
        ("2024-03-15 → ISO", check_date_iso),
        ("March 15, 44 BC → year=-44", check_date_bc),
        ("circa 500 AD → approximate=True", check_date_circa),
    ]:
        total += 1
        if check(label, fn):
            passed += 1
        else:
            failed += 1

    # ----------------------------------------------------------------
    # 5. Graph compilation with web search
    # ----------------------------------------------------------------
    print("\n5. Graph compilation with web search")

    def check_graph_without_web():
        from workbench.core.types import ModelResponse
        from workbench.graphs.rag_graph import build_rag_graph

        class FakeSearch:
            def execute(self, **kw):
                return {"chunks": [], "count": 0}

        class FakeOpen:
            def execute(self, **kw):
                return {"found": False, "chunk": None}

        class FakeModel:
            def generate(self, msgs, **kw):
                return ModelResponse(
                    text="test", tokens_in=1, tokens_out=1, latency_ms=1.0
                )

        g = build_rag_graph(
            search_tool=FakeSearch(),
            open_chunk_tool=FakeOpen(),
            model=FakeModel(),
        )
        assert hasattr(g, "invoke")

    def check_graph_with_web():
        from workbench.core.types import ModelResponse
        from workbench.graphs.rag_graph import build_rag_graph

        class FakeSearch:
            def execute(self, **kw):
                return {"chunks": [], "count": 0}

        class FakeOpen:
            def execute(self, **kw):
                return {"found": False, "chunk": None}

        class FakeModel:
            def generate(self, msgs, **kw):
                return ModelResponse(
                    text="test", tokens_in=1, tokens_out=1, latency_ms=1.0
                )

        class FakeWebSearch:
            def execute(self, **kw):
                return {"results": [], "count": 0, "provider": "fake"}

        g = build_rag_graph(
            search_tool=FakeSearch(),
            open_chunk_tool=FakeOpen(),
            model=FakeModel(),
            web_search_tool=FakeWebSearch(),
            confidence_threshold=2,
        )
        assert hasattr(g, "invoke")

    for label, fn in [
        ("RAG graph without web search (backward compat)", check_graph_without_web),
        ("RAG graph WITH web search tool", check_graph_with_web),
    ]:
        total += 1
        if check(label, fn):
            passed += 1
        else:
            failed += 1

    # ----------------------------------------------------------------
    # 6. LangChain adapter
    # ----------------------------------------------------------------
    print("\n6. LangChain tool adapter")

    def check_langchain_adapter():
        from workbench.tools.utility.calculator import CalculatorTool

        t = CalculatorTool()
        lc = t.to_langchain_tool()
        assert lc.name == "calculator"
        result_str = lc.invoke({"expression": "2 + 2"})
        assert "4" in result_str

    for label, fn in [
        ("CalculatorTool.to_langchain_tool()", check_langchain_adapter),
    ]:
        total += 1
        if check(label, fn):
            passed += 1
        else:
            failed += 1

    # ----------------------------------------------------------------
    # 7. Config
    # ----------------------------------------------------------------
    print("\n7. Config file")

    def check_config():
        import yaml

        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "defaults.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "web_search" in cfg, "Missing web_search section"
        assert cfg["web_search"]["provider"] in ("duckduckgo", "tavily")
        assert "source_fetch" in cfg, "Missing source_fetch section"

    for label, fn in [
        ("defaults.yaml has web_search + source_fetch sections", check_config),
    ]:
        total += 1
        if check(label, fn):
            passed += 1
        else:
            failed += 1

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("🎉  Day 5 setup is complete!")
    else:
        print("⚠️   Some checks failed — review the output above.")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
