"""
Tests for Day 8 cleanup.

Validates:
    - GraphRunner and run_graph are importable from systems
    - agents.__init__ exports state types (AgentState, OpenedChunk, etc.)
    - Legacy WikiExtractor functions are removed from ingest_wikipedia
    - Legacy orchestration (AgentLoop, policies, ChatbotSystem) is removed
    - Graph result → AgentState adapter works
    - AgentState (still in agents.state) is unchanged
"""

from __future__ import annotations

import pytest

# ------------------------------------------------------------------
# 1. Systems runner
# ------------------------------------------------------------------


class TestSystemsRunner:
    """Verify the unified runner module is importable."""

    def test_graph_runner_importable(self):
        from workbench.systems.runner import GraphRunner

        assert GraphRunner is not None

    def test_run_graph_importable(self):
        from workbench.systems.runner import run_graph

        assert callable(run_graph)

    def test_systems_init_exports(self):
        from workbench.systems import GraphRunner, run_graph

        assert GraphRunner is not None
        assert callable(run_graph)

    def test_supported_graphs(self):
        from workbench.systems.runner import SUPPORTED_GRAPHS

        assert "rag_graph" in SUPPORTED_GRAPHS
        assert "researcher_graph" in SUPPORTED_GRAPHS
        assert "supervisor_graph" in SUPPORTED_GRAPHS

    def test_graph_runner_rejects_unknown(self):
        from workbench.systems.runner import GraphRunner

        runner = GraphRunner.__new__(GraphRunner)
        runner._components = None
        with pytest.raises(ValueError, match="Unknown graph"):
            runner.build_graph("nonexistent_graph")


# ------------------------------------------------------------------
# 2. Wikipedia ingest cleanup
# ------------------------------------------------------------------


class TestIngestCleanup:
    """Verify WikiExtractor functions are removed and primary path exists."""

    def test_primary_path_exists(self):
        from workbench.data_build.ingest_wikipedia import ingest_wikipedia_dump

        assert callable(ingest_wikipedia_dump)

    def test_iter_articles_exists(self):
        from workbench.data_build.ingest_wikipedia import iter_articles_from_dump

        assert callable(iter_articles_from_dump)

    def test_strip_wiki_markup_exists(self):
        from workbench.data_build.ingest_wikipedia import strip_wiki_markup

        assert callable(strip_wiki_markup)

    def test_wikiextractor_functions_removed(self):
        import workbench.data_build.ingest_wikipedia as mod

        assert not hasattr(mod, "parse_extracted_file"), (
            "parse_extracted_file should be removed"
        )
        assert not hasattr(mod, "iter_extracted_dir"), (
            "iter_extracted_dir should be removed"
        )
        assert not hasattr(mod, "ingest_wikipedia"), (
            "ingest_wikipedia (legacy) should be removed"
        )


# ------------------------------------------------------------------
# 3. Legacy orchestration fully removed
# ------------------------------------------------------------------


class TestLegacyRemoved:
    """Verify legacy orchestration code is deleted."""

    def test_no_legacy_package(self):
        """workbench.legacy should not be importable."""
        with pytest.raises(ImportError):
            import workbench.legacy  # noqa: F401

    def test_no_agents_loops_module(self):
        """workbench.agents.loops should not be importable."""
        with pytest.raises(ImportError):
            from workbench.agents.loops import AgentLoop  # noqa: F401

    def test_no_agents_policies_module(self):
        """workbench.agents.policies should not be importable."""
        with pytest.raises(ImportError):
            from workbench.agents.policies import ScriptedPolicy  # noqa: F401

    def test_no_chatbot_system(self):
        """workbench.systems.chatbot should not be importable."""
        with pytest.raises(ImportError):
            from workbench.systems.chatbot import ChatbotSystem  # noqa: F401

    def test_agents_init_exports_state_only(self):
        """agents.__init__ exports state types, not legacy loop/policies."""
        import workbench.agents as agents_pkg

        # State types should be present
        assert hasattr(agents_pkg, "AgentState")
        assert hasattr(agents_pkg, "OpenedChunk")
        assert hasattr(agents_pkg, "RetrievedChunkSummary")
        assert hasattr(agents_pkg, "TimelineItem")

        # Legacy should NOT be present
        assert not hasattr(agents_pkg, "AgentLoop")
        assert not hasattr(agents_pkg, "ScriptedPolicy")
        assert not hasattr(agents_pkg, "ResearcherPolicy")


# ------------------------------------------------------------------
# 4. Graph result → AgentState adapter
# ------------------------------------------------------------------


class TestGraphResultAdapter:
    """Test converting a graph result dict to AgentState."""

    def test_converts_basic_result(self):
        from workbench.agents.state import (
            AgentState,
            OpenedChunk,
            RetrievedChunkSummary,
            TimelineItem,
        )

        result = {
            "question": "test?",
            "retrieved": [
                {"chunk_id": "c1", "score": 0.9, "title": "T", "snippet": "S"},
            ],
            "opened": [
                {"chunk_id": "c1", "title": "T", "section": "S", "text": "Text."},
            ],
            "timeline": [
                {"date": "100 BC", "event": "E", "supporting_chunk_ids": ["c1"]},
            ],
            "answer_text": "Answer [c1].",
            "citations": ["c1"],
            "tokens_in": 100,
            "tokens_out": 50,
            "model_latency_ms": 500.0,
        }

        # Build AgentState the same way the eval script does
        state = AgentState(question="test?")
        for r in result["retrieved"]:
            state.retrieved.append(RetrievedChunkSummary(**r))
        for o in result["opened"]:
            state.opened.append(OpenedChunk(**o))
        for t in result["timeline"]:
            state.timeline.append(TimelineItem(**t))
        state.answer_text = result["answer_text"]
        state.citations = result["citations"]
        state.tokens_in = result["tokens_in"]
        state.tokens_out = result["tokens_out"]
        state.model_latency_ms = result["model_latency_ms"]
        state.done = True

        assert len(state.retrieved) == 1
        assert len(state.opened) == 1
        assert len(state.timeline) == 1
        assert state.answer_text == "Answer [c1]."
        assert state.citations == ["c1"]
        assert state.tokens_in == 100

    def test_handles_empty_result(self):
        from workbench.agents.state import AgentState

        state = AgentState(question="test?")
        state.done = True
        state.error = "some error"

        assert state.question == "test?"
        assert len(state.retrieved) == 0
        assert state.error == "some error"


# ------------------------------------------------------------------
# 5. AgentState unchanged
# ------------------------------------------------------------------


class TestAgentStateUnchanged:
    """AgentState should still be in agents.state, not moved."""

    def test_importable_from_agents_state(self):
        from workbench.agents.state import AgentState

        assert AgentState is not None

    def test_importable_from_agents_init(self):
        from workbench.agents import AgentState

        assert AgentState is not None

    def test_has_expected_fields(self):
        from workbench.agents.state import AgentState

        state = AgentState(question="test?")
        assert hasattr(state, "question")
        assert hasattr(state, "retrieved")
        assert hasattr(state, "opened")
        assert hasattr(state, "timeline")
        assert hasattr(state, "answer_text")
        assert hasattr(state, "citations")
        assert hasattr(state, "step")
        assert hasattr(state, "done")
        assert hasattr(state, "error")
        assert hasattr(state, "tokens_in")
        assert hasattr(state, "tokens_out")
        assert hasattr(state, "model_latency_ms")

    def test_summary_method(self):
        from workbench.agents.state import AgentState

        state = AgentState(question="test?")
        s = state.summary()
        assert isinstance(s, dict)
        assert "question" in s
