"""
Tests for LangGraph integration (Day 0).

Verifies that:
- LangGraph imports successfully
- The smoke graph compiles and runs
- State transitions produce expected output
- The graph has the expected node structure
"""

from __future__ import annotations

from scripts.langgraph_smoke import SmallState, build_graph, greet, shout


class TestLangGraphImports:
    """Verify LangGraph packages are importable."""

    def test_langgraph_imports(self):
        from langgraph.graph import END, StateGraph

        assert StateGraph is not None
        assert END is not None

    def test_langchain_core_imports(self):
        import langchain_core

        assert langchain_core is not None


class TestSmallStateNodes:
    """Unit tests for the individual node functions."""

    def test_greet_sets_message(self):
        state: SmallState = {"message": "hi", "steps": 0}
        result = greet(state)
        assert "Hello from LangGraph!" in result["message"]
        assert "hi" in result["message"]

    def test_greet_increments_steps(self):
        state: SmallState = {"message": "hi", "steps": 0}
        result = greet(state)
        assert result["steps"] == 1

    def test_shout_uppercases(self):
        state: SmallState = {"message": "hello world", "steps": 1}
        result = shout(state)
        assert result["message"] == "HELLO WORLD"

    def test_shout_increments_steps(self):
        state: SmallState = {"message": "hello", "steps": 1}
        result = shout(state)
        assert result["steps"] == 2


class TestSmokeGraph:
    """Integration tests for the compiled graph."""

    def test_graph_compiles(self):
        app = build_graph()
        assert app is not None

    def test_graph_invoke_produces_result(self):
        app = build_graph()
        result = app.invoke({"message": "test", "steps": 0})
        assert isinstance(result, dict)
        assert "message" in result
        assert "steps" in result

    def test_graph_runs_both_nodes(self):
        app = build_graph()
        result = app.invoke({"message": "test", "steps": 0})
        assert result["steps"] == 2

    def test_graph_output_is_uppercased_greeting(self):
        app = build_graph()
        result = app.invoke({"message": "workbench", "steps": 0})
        assert result["message"] == "HELLO FROM LANGGRAPH! (INPUT WAS: WORKBENCH)"

    def test_graph_is_deterministic(self):
        app = build_graph()
        r1 = app.invoke({"message": "abc", "steps": 0})
        r2 = app.invoke({"message": "abc", "steps": 0})
        assert r1 == r2
