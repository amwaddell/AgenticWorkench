"""
Tests for the agent loop's max_steps safety guard.

Uses fake tools and a NeverStopPolicy to verify that the loop
terminates after max_steps even if the policy keeps requesting actions.

No model server, no LanceDB, no heavy imports needed.
"""

from __future__ import annotations

from typing import Any

from workbench.agents.loops import AgentLoop
from workbench.agents.policies import NeverStopPolicy
from workbench.agents.state import AgentState

# --- Fakes -------------------------------------------------------------------


class FakeSearchTool:
    """Search tool that always returns one dummy result."""

    name = "search_wikipedia"
    description = "fake"

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "chunks": [
                {
                    "chunk_id": "fake_001",
                    "score": 1.0,
                    "title": "Fake Article",
                    "snippet": "This is a fake chunk for testing.",
                }
            ],
            "count": 1,
            "duration_ms": 0.1,
        }


class FakeOpenChunkTool:
    """Open-chunk tool that always returns a dummy chunk."""

    name = "open_chunk"
    description = "fake"

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "found": True,
            "chunk": {
                "chunk_id": kwargs.get("chunk_id", "fake_001"),
                "document_id": "doc_fake",
                "title": "Fake Article",
                "section": None,
                "text": "Fake chunk text for testing.",
            },
            "text_length": 28,
            "duration_ms": 0.1,
        }


class FakeModel:
    """Model that returns a canned response."""

    def generate(self, messages: list[dict[str, str]], **settings: Any) -> Any:
        from workbench.core.types import ModelResponse

        return ModelResponse(
            text="This is a fake answer.",
            tokens_in=10,
            tokens_out=5,
            latency_ms=1.0,
        )


# --- Tests -------------------------------------------------------------------


class TestAgentLoopLimits:
    """Tests for max_steps enforcement."""

    def test_max_steps_stops_never_stop_policy(self) -> None:
        """
        NeverStopPolicy always returns SEARCH.  The loop must stop
        after max_steps regardless.
        """
        loop = AgentLoop(
            policy=NeverStopPolicy(),
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            max_steps=5,
        )

        state = AgentState(question="test question")
        result = loop.run(state)

        assert result.done is True
        assert result.step == 5
        assert result.error is not None
        assert "max_steps" in result.error

    def test_max_steps_1_stops_immediately(self) -> None:
        """With max_steps=1, only one action should execute."""
        loop = AgentLoop(
            policy=NeverStopPolicy(),
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            max_steps=1,
        )

        state = AgentState(question="test")
        result = loop.run(state)

        assert result.done is True
        assert result.step == 1

    def test_max_steps_3_executes_exactly_3(self) -> None:
        """Loop should execute exactly max_steps iterations."""
        loop = AgentLoop(
            policy=NeverStopPolicy(),
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            max_steps=3,
        )

        state = AgentState(question="test")
        result = loop.run(state)

        assert result.step == 3

    def test_scripted_policy_completes_before_max_steps(self) -> None:
        """
        ScriptedPolicy should finish in 3 steps (search, open, generate),
        well before max_steps=10.
        """
        from workbench.agents.policies import ScriptedPolicy

        loop = AgentLoop(
            policy=ScriptedPolicy(open_top_n=1, search_top_k=1),
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            max_steps=10,
        )

        state = AgentState(question="test question")
        result = loop.run(state)

        assert result.done is True
        assert result.step <= 4  # search + open + generate + stop check
        assert result.answer_text == "This is a fake answer."
        assert result.error is None

    def test_error_in_tool_stops_loop(self) -> None:
        """If a tool raises, the loop should catch it, set error, and stop."""

        class BrokenSearchTool:
            name = "search_wikipedia"
            description = "broken"

            def execute(self, **kwargs: Any) -> dict:
                raise RuntimeError("connection refused")

        loop = AgentLoop(
            policy=NeverStopPolicy(),
            search_tool=BrokenSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            max_steps=10,
        )

        state = AgentState(question="test")
        result = loop.run(state)

        assert result.done is True
        assert result.error is not None
        assert "connection refused" in result.error
        assert result.step == 1  # failed on first step
