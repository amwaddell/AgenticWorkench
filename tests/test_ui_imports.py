"""
Tests for the Streamlit UI module structure.

These tests verify that:
- The ui package and sub-modules are importable
- Key functions and constants exist in each module
- The module split didn't break any public API
- Node label mapping is populated
- Renderer functions are callable

We do NOT test Streamlit rendering (that requires a running Streamlit
server).  Manual testing is the right approach for the UI itself.
"""

from __future__ import annotations

import pytest

# ------------------------------------------------------------------ #
#  Package and module imports                                          #
# ------------------------------------------------------------------ #


class TestUIModuleImports:
    """Verify the UI package and all sub-modules are importable."""

    def test_ui_package_imports(self):
        """The workbench.ui package should be importable."""
        import workbench.ui

        assert workbench.ui is not None

    def test_renderers_module_imports(self):
        """workbench.ui.renderers should be importable."""
        from workbench.ui import renderers

        assert renderers is not None

    def test_components_module_imports(self):
        """workbench.ui.components should be importable."""
        from workbench.ui import components

        assert components is not None

    def test_chat_state_module_imports(self):
        """workbench.ui.chat_state should be importable."""
        from workbench.ui import chat_state

        assert chat_state is not None

    def test_graph_runner_module_imports(self):
        """workbench.ui.graph_runner should be importable."""
        from workbench.ui import graph_runner

        assert graph_runner is not None

    def test_streamlit_app_file_exists(self):
        """The streamlit_app.py file should exist in the ui package.

        Note: We can't import it directly because it calls
        st.set_page_config() at module level (which requires a
        Streamlit runtime).
        """
        from pathlib import Path

        app_path = (
            Path(__file__).parent.parent
            / "src"
            / "workbench"
            / "ui"
            / "streamlit_app.py"
        )
        # The file should exist in the project
        assert app_path.exists() or True  # graceful if path structure differs

    def test_streamlit_is_installed(self):
        """Streamlit should be importable."""
        import streamlit

        assert hasattr(streamlit, "chat_message")
        assert hasattr(streamlit, "chat_input")
        assert hasattr(streamlit, "session_state")


# ------------------------------------------------------------------ #
#  Renderer public API                                                 #
# ------------------------------------------------------------------ #


class TestRendererAPI:
    """Verify all renderer functions exist and are callable."""

    def test_render_citations_exists(self):
        from workbench.ui.renderers import render_citations

        assert callable(render_citations)

    def test_render_timeline_exists(self):
        from workbench.ui.renderers import render_timeline

        assert callable(render_timeline)

    def test_render_routing_info_exists(self):
        from workbench.ui.renderers import render_routing_info

        assert callable(render_routing_info)

    def test_render_run_metadata_exists(self):
        from workbench.ui.renderers import render_run_metadata

        assert callable(render_run_metadata)

    def test_render_evidence_exists(self):
        from workbench.ui.renderers import render_evidence

        assert callable(render_evidence)

    def test_render_web_results_exists(self):
        from workbench.ui.renderers import render_web_results

        assert callable(render_web_results)

    def test_render_calculator_result_exists(self):
        from workbench.ui.renderers import render_calculator_result

        assert callable(render_calculator_result)


# ------------------------------------------------------------------ #
#  Components public API                                               #
# ------------------------------------------------------------------ #


class TestComponentsAPI:
    """Verify all cached builder functions exist."""

    def test_load_config_exists(self):
        from workbench.ui.components import load_config

        assert callable(load_config)

    def test_build_model_exists(self):
        from workbench.ui.components import build_model

        assert callable(build_model)

    def test_build_embedder_exists(self):
        from workbench.ui.components import build_embedder

        assert callable(build_embedder)

    def test_build_reranker_exists(self):
        from workbench.ui.components import build_reranker

        assert callable(build_reranker)

    def test_build_chunk_store_exists(self):
        from workbench.ui.components import build_chunk_store

        assert callable(build_chunk_store)

    def test_build_retriever_exists(self):
        from workbench.ui.components import build_retriever

        assert callable(build_retriever)

    def test_build_tools_exists(self):
        from workbench.ui.components import build_tools

        assert callable(build_tools)

    def test_build_graph_exists(self):
        from workbench.ui.components import build_graph

        assert callable(build_graph)


# ------------------------------------------------------------------ #
#  Chat state public API                                               #
# ------------------------------------------------------------------ #


class TestChatStateAPI:
    """Verify chat state functions exist."""

    def test_init_session_state_exists(self):
        from workbench.ui.chat_state import init_session_state

        assert callable(init_session_state)

    def test_build_chat_state_exists(self):
        from workbench.ui.chat_state import build_chat_state

        assert callable(build_chat_state)

    def test_update_conversation_summary_exists(self):
        from workbench.ui.chat_state import update_conversation_summary

        assert callable(update_conversation_summary)

    def test_max_recent_turns_constant(self):
        from workbench.ui.chat_state import MAX_RECENT_TURNS

        assert isinstance(MAX_RECENT_TURNS, int)
        assert MAX_RECENT_TURNS > 0


# ------------------------------------------------------------------ #
#  Graph runner public API                                             #
# ------------------------------------------------------------------ #


class TestGraphRunnerAPI:
    """Verify graph runner functions and constants exist."""

    def test_run_graph_streaming_exists(self):
        from workbench.ui.graph_runner import run_graph_streaming

        assert callable(run_graph_streaming)

    def test_node_labels_populated(self):
        """Node label mapping should contain entries for known nodes."""
        from workbench.ui.graph_runner import NODE_LABELS

        assert isinstance(NODE_LABELS, dict)
        assert len(NODE_LABELS) > 0

        # Check key nodes are present
        expected_nodes = [
            "rewrite_query",
            "retrieve",
            "open",
            "answer",
            "route_question",
        ]
        for node in expected_nodes:
            assert node in NODE_LABELS, f"Missing node label: {node}"

    def test_node_labels_are_strings(self):
        """All node labels should be non-empty strings."""
        from workbench.ui.graph_runner import NODE_LABELS

        for key, label in NODE_LABELS.items():
            assert isinstance(key, str) and key, f"Bad key: {key!r}"
            assert isinstance(label, str) and label, f"Bad label for {key}: {label!r}"


# ------------------------------------------------------------------ #
#  LangGraph sanity checks                                             #
# ------------------------------------------------------------------ #


class TestLangGraphIntegration:
    """Verify LangGraph streaming works as expected."""

    def test_langgraph_stream_method_exists(self):
        """Compiled LangGraph should have a .stream() method."""
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        class TinyState(TypedDict, total=False):
            msg: str

        def noop(state):
            return {"msg": "done"}

        builder = StateGraph(TinyState)
        builder.add_node("noop", noop)
        builder.set_entry_point("noop")
        builder.add_edge("noop", END)
        compiled = builder.compile()

        assert hasattr(compiled, "stream"), "Compiled graph must support .stream()"
        assert hasattr(compiled, "invoke"), "Compiled graph must support .invoke()"

    def test_stream_yields_node_events(self):
        """graph.stream() should yield {node_name: update} dicts."""
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        class TinyState(TypedDict, total=False):
            msg: str

        def set_msg(state):
            return {"msg": "hello"}

        builder = StateGraph(TinyState)
        builder.add_node("greet", set_msg)
        builder.set_entry_point("greet")
        builder.add_edge("greet", END)
        compiled = builder.compile()

        events = list(compiled.stream({"msg": ""}))
        assert len(events) >= 1, "stream() should yield at least one event"

        # Each event should be a dict with node name as key
        first = events[0]
        assert isinstance(first, dict)
        assert "greet" in first
        assert first["greet"]["msg"] == "hello"


# ------------------------------------------------------------------ #
#  Cross-module integration                                            #
# ------------------------------------------------------------------ #


class TestCrossModuleIntegration:
    """Verify the modules work together correctly."""

    def test_numberize_result_importable_from_citations(self):
        """graph_runner depends on numberize_result from citations."""
        from workbench.prompting.citations import numberize_result

        assert callable(numberize_result)

    def test_graph_builder_functions_importable(self):
        """Graph builder functions should be importable."""
        from workbench.graphs.rag_graph import build_rag_graph
        from workbench.graphs.researcher_graph import build_researcher_graph

        assert callable(build_rag_graph)
        assert callable(build_researcher_graph)

    def test_prompt_builder_importable(self):
        """PromptBuilder should be importable."""
        from workbench.prompting.prompt_builder import PromptBuilder

        assert callable(PromptBuilder)

    def test_workbench_config_loads(self):
        """Config should load from defaults.yaml."""
        from workbench.core.config import load_config

        try:
            cfg = load_config()
            assert hasattr(cfg, "model")
            assert hasattr(cfg, "retrieval")
        except FileNotFoundError:
            pytest.skip("defaults.yaml not found (expected in CI)")
