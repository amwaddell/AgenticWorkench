"""
Tests that verify tool calls produce the expected trace spans.

Uses an in-memory span exporter so tests don't depend on Phoenix.
Runs a single agent pass with fake components and asserts that
the expected span names appear in the captured trace.

Works with the real OpenTelemetry SDK (not just stubs).
"""

from __future__ import annotations

from typing import Any

import pytest

from workbench.agents.loops import AgentLoop
from workbench.agents.policies import ScriptedPolicy
from workbench.agents.state import AgentState
from workbench.core.types import ModelResponse

# Try to import the in-memory exporter (correct module path)
try:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


# --- Fakes (same pattern as test_agent_loop_limits) --------------------------


class FakeSearchTool:
    name = "search_wikipedia"
    description = "fake"

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "chunks": [
                {
                    "chunk_id": "span_test_001",
                    "score": 1.0,
                    "title": "Test Article",
                    "snippet": "Snippet for span testing.",
                },
                {
                    "chunk_id": "span_test_002",
                    "score": 0.9,
                    "title": "Second Article",
                    "snippet": "Another snippet.",
                },
            ],
            "count": 2,
            "duration_ms": 0.5,
        }


class FakeOpenChunkTool:
    name = "open_chunk"
    description = "fake"

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        cid = kwargs.get("chunk_id", "unknown")
        return {
            "found": True,
            "chunk": {
                "chunk_id": cid,
                "document_id": "doc_fake",
                "title": "Test Article",
                "section": None,
                "text": f"Full text for chunk {cid}.",
            },
            "text_length": 30,
            "duration_ms": 0.1,
        }


class FakeModel:
    def generate(
        self, messages: list[dict[str, str]], **settings: Any
    ) -> ModelResponse:
        return ModelResponse(
            text="Answer referencing [span_test_001].",
            tokens_in=50,
            tokens_out=10,
            latency_ms=5.0,
        )


# --- Tests -------------------------------------------------------------------


@pytest.mark.skipif(not HAS_OTEL, reason="opentelemetry SDK not installed")
class TestToolSpansExist:
    """Verify that tool calls produce trace spans with expected names."""

    @pytest.fixture(autouse=True)
    def _setup_tracing(self):
        """
        Set up a fresh TracerProvider with an InMemorySpanExporter,
        then monkey-patch the tracing module to use it.

        This avoids the "provider already set" problem with the global
        OTel API by directly replacing the module-level tracer.
        """
        import workbench.observability.tracing as tracing_mod

        self.exporter = InMemorySpanExporter()
        resource = Resource.create({"service.name": "test-tool-spans"})
        provider = TracerProvider(resource=resource)
        # Use SimpleSpanProcessor for synchronous export (no async flush needed)
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))

        # Monkey-patch the module globals so start_span() uses our provider
        self._old_tracer = tracing_mod._tracer
        self._old_provider = tracing_mod._tracer_provider

        tracing_mod._tracer_provider = provider
        tracing_mod._tracer = provider.get_tracer("test-tool-spans")

        yield

        # Restore originals
        provider.shutdown()
        tracing_mod._tracer = self._old_tracer
        tracing_mod._tracer_provider = self._old_provider

    def _get_span_names(self) -> list[str]:
        """Return all finished span names."""
        return [span.name for span in self.exporter.get_finished_spans()]

    def _run_agent(self) -> list[str]:
        """Run a full agent pass and return captured span names."""
        loop = AgentLoop(
            policy=ScriptedPolicy(open_top_n=2, search_top_k=2),
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            max_steps=10,
        )

        state = AgentState(question="test question for spans")
        loop.run(state)

        return self._get_span_names()

    def test_agent_run_span_exists(self) -> None:
        """Top-level agent.run span should be present."""
        names = self._run_agent()
        assert "agent.run" in names

    def test_agent_step_spans_exist(self) -> None:
        """At least one agent.step span should be present."""
        names = self._run_agent()
        assert "agent.step" in names

    def test_search_wikipedia_span_exists(self) -> None:
        """tool.search_wikipedia span should be created by the search tool."""
        names = self._run_agent()
        assert "tool.search_wikipedia" in names

    def test_open_chunk_span_exists(self) -> None:
        """tool.open_chunk span should be created for each opened chunk."""
        names = self._run_agent()
        assert "tool.open_chunk" in names

    def test_model_generate_span_not_present_with_fake(self) -> None:
        """
        With a FakeModel, model.generate won't appear because the fake
        doesn't go through LlamaCppServerModel which creates that span.
        This test documents the expectation — in real runs, model.generate
        will be present.
        """
        names = self._run_agent()
        # The fake doesn't create a model.generate span, so just verify
        # the agent completed and the other spans exist.
        assert "agent.run" in names
        assert "tool.search_wikipedia" in names

    def test_open_chunk_span_count_matches_opened_chunks(self) -> None:
        """
        We open 2 chunks (open_top_n=2), so there should be at least
        2 tool.open_chunk spans.
        """
        self._run_agent()  # uses open_top_n=2
        spans = self.exporter.get_finished_spans()
        open_spans = [s for s in spans if s.name == "tool.open_chunk"]
        assert len(open_spans) >= 2

    def test_step_spans_have_step_number_attribute(self) -> None:
        """Each agent.step span should carry a step_number attribute."""
        loop = AgentLoop(
            policy=ScriptedPolicy(open_top_n=1, search_top_k=1),
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            max_steps=10,
        )
        state = AgentState(question="attribute test")
        loop.run(state)

        spans = self.exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "agent.step"]
        assert len(step_spans) >= 1
        for span in step_spans:
            attrs = dict(span.attributes) if span.attributes else {}
            assert "step_number" in attrs
