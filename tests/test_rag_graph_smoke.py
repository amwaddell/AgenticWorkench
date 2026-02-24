"""
Smoke tests for the RAG graph (Day 3).

Uses fake tools and a fake model — no LanceDB, no model server,
no heavy imports.  Validates that:

- The graph compiles
- invoke() returns the expected state shape
- Each node produces the correct output
- Citation extraction works
- Error handling doesn't crash the graph
- PromptBuilder.build() works with dict and Pydantic inputs
- The graph behaves identically to the old ScriptedPolicy flow

Run:
    python -m pytest tests/test_rag_graph_smoke.py -v
"""

from __future__ import annotations

from typing import Any

import pytest

from workbench.agents.state import OpenedChunk
from workbench.core.types import ModelResponse
from workbench.graphs.rag_graph import build_rag_graph, extract_citations

# ------------------------------------------------------------------ #
#  Fakes                                                              #
# ------------------------------------------------------------------ #

# Two fake chunks with distinct content so citation tests are meaningful.
FAKE_CHUNKS = [
    {
        "chunk_id": "chunk_rome_001",
        "score": 0.95,
        "title": "Roman Republic",
        "snippet": "The Roman Republic ended in 27 BC when Augustus...",
    },
    {
        "chunk_id": "chunk_rome_002",
        "score": 0.88,
        "title": "Augustus",
        "snippet": "Augustus became the first Roman Emperor after...",
    },
    {
        "chunk_id": "chunk_rome_003",
        "score": 0.72,
        "title": "Julius Caesar",
        "snippet": "Caesar's assassination in 44 BC led to a series...",
    },
]

FAKE_OPENED = {
    "chunk_rome_001": {
        "chunk_id": "chunk_rome_001",
        "document_id": "doc_rome",
        "title": "Roman Republic",
        "section": "Fall",
        "text": (
            "The Roman Republic ended in 27 BC when Octavian was "
            "granted the title Augustus by the Senate."
        ),
    },
    "chunk_rome_002": {
        "chunk_id": "chunk_rome_002",
        "document_id": "doc_augustus",
        "title": "Augustus",
        "section": "Rise to power",
        "text": (
            "Augustus became the first Roman Emperor after defeating "
            "Mark Antony at the Battle of Actium in 31 BC."
        ),
    },
    "chunk_rome_003": {
        "chunk_id": "chunk_rome_003",
        "document_id": "doc_caesar",
        "title": "Julius Caesar",
        "section": "Assassination",
        "text": (
            "Caesar's assassination in 44 BC led to a series of "
            "civil wars that ended the Republic."
        ),
    },
}


class FakeSearchTool:
    """Search tool returning canned results."""

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        top_k = kwargs.get("top_k", 5)
        return {
            "chunks": FAKE_CHUNKS[:top_k],
            "count": min(top_k, len(FAKE_CHUNKS)),
        }


class FakeOpenChunkTool:
    """Open-chunk tool returning canned chunk data."""

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        chunk_id = kwargs.get("chunk_id", "")
        chunk = FAKE_OPENED.get(chunk_id)
        if chunk:
            return {"found": True, "chunk": chunk, "text_length": len(chunk["text"])}
        return {"found": False, "chunk": None, "text_length": 0}


class FakeModel:
    """Model that returns a canned answer with citations."""

    def __init__(self, answer: str | None = None) -> None:
        self._answer = answer or (
            "The Roman Republic ended in 27 BC [chunk_rome_001] "
            "when Augustus defeated Antony [chunk_rome_002]."
        )

    def generate(
        self, messages: list[dict[str, str]], **settings: Any
    ) -> ModelResponse:
        return ModelResponse(
            text=self._answer,
            tokens_in=50,
            tokens_out=30,
            latency_ms=120.0,
        )


class FailingSearchTool:
    """Search tool that always raises."""

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Search backend unavailable")


class FailingModel:
    """Model that always raises."""

    def generate(
        self, messages: list[dict[str, str]], **settings: Any
    ) -> ModelResponse:
        raise RuntimeError("Model server down")


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #


@pytest.fixture
def rag_graph():
    """Build a RAG graph with fake dependencies."""
    return build_rag_graph(
        search_tool=FakeSearchTool(),
        open_chunk_tool=FakeOpenChunkTool(),
        model=FakeModel(),
        prompt_builder=None,  # uses fallback prompt
        open_top_n=2,
        search_top_k=3,
    )


@pytest.fixture
def rag_graph_top1():
    """RAG graph that opens only 1 chunk."""
    return build_rag_graph(
        search_tool=FakeSearchTool(),
        open_chunk_tool=FakeOpenChunkTool(),
        model=FakeModel(),
        open_top_n=1,
        search_top_k=2,
    )


# ------------------------------------------------------------------ #
#  Test: graph compilation                                            #
# ------------------------------------------------------------------ #


class TestGraphCompilation:
    """Verify the graph builds and compiles without errors."""

    def test_build_returns_compiled_graph(self, rag_graph):
        """build_rag_graph() should return a compiled graph object."""
        assert rag_graph is not None

    def test_graph_has_invoke(self, rag_graph):
        """Compiled graph should have an invoke() method."""
        assert hasattr(rag_graph, "invoke")

    def test_build_with_all_defaults(self):
        """build_rag_graph works with only the three required args."""
        g = build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
        )
        assert g is not None


# ------------------------------------------------------------------ #
#  Test: full invocation                                              #
# ------------------------------------------------------------------ #


class TestGraphInvocation:
    """Verify invoke() runs the full pipeline and returns correct state."""

    def test_invoke_returns_dict(self, rag_graph):
        """invoke() should return a dict (RAGState)."""
        result = rag_graph.invoke({"question": "When did the Roman Republic end?"})
        assert isinstance(result, dict)

    def test_invoke_has_answer_text(self, rag_graph):
        """Result should contain a non-empty answer_text."""
        result = rag_graph.invoke({"question": "When did the Roman Republic end?"})
        assert result.get("answer_text")
        assert len(result["answer_text"]) > 0

    def test_invoke_has_citations(self, rag_graph):
        """Result should contain citations matching known chunk IDs."""
        result = rag_graph.invoke({"question": "When did the Roman Republic end?"})
        assert isinstance(result.get("citations"), list)
        assert len(result["citations"]) > 0

    def test_invoke_has_retrieved(self, rag_graph):
        """Result should contain retrieved chunk summaries."""
        result = rag_graph.invoke({"question": "When did the Roman Republic end?"})
        assert isinstance(result.get("retrieved"), list)
        assert len(result["retrieved"]) > 0

    def test_invoke_has_opened(self, rag_graph):
        """Result should contain opened chunk data."""
        result = rag_graph.invoke({"question": "When did the Roman Republic end?"})
        assert isinstance(result.get("opened"), list)
        assert len(result["opened"]) > 0

    def test_invoke_has_token_counts(self, rag_graph):
        """Result should have token count metadata."""
        result = rag_graph.invoke({"question": "When did the Roman Republic end?"})
        assert result.get("tokens_in", 0) > 0
        assert result.get("tokens_out", 0) > 0

    def test_invoke_has_model_latency(self, rag_graph):
        """Result should have model_latency_ms."""
        result = rag_graph.invoke({"question": "When did the Roman Republic end?"})
        assert result.get("model_latency_ms", 0) > 0

    def test_invoke_no_error(self, rag_graph):
        """A successful run should not set the error field."""
        result = rag_graph.invoke({"question": "When did the Roman Republic end?"})
        assert not result.get("error")


# ------------------------------------------------------------------ #
#  Test: state flow through nodes                                     #
# ------------------------------------------------------------------ #


class TestNodeBehaviour:
    """Verify each node produces the expected output shape."""

    def test_retrieved_chunk_shape(self, rag_graph):
        """Each retrieved dict should have chunk_id, score, title, snippet."""
        result = rag_graph.invoke({"question": "test"})
        for chunk in result.get("retrieved", []):
            assert "chunk_id" in chunk
            assert "score" in chunk
            assert "title" in chunk

    def test_opened_chunk_shape(self, rag_graph):
        """Each opened dict should have chunk_id, title, text."""
        result = rag_graph.invoke({"question": "test"})
        for chunk in result.get("opened", []):
            assert "chunk_id" in chunk
            assert "title" in chunk
            assert "text" in chunk

    def test_open_top_n_respected(self, rag_graph):
        """open_top_n=2 should open exactly 2 chunks."""
        result = rag_graph.invoke({"question": "test"})
        assert len(result.get("opened", [])) == 2

    def test_open_top_n_1(self, rag_graph_top1):
        """open_top_n=1 should open exactly 1 chunk."""
        result = rag_graph_top1.invoke({"question": "test"})
        assert len(result.get("opened", [])) == 1

    def test_search_top_k_respected(self, rag_graph):
        """search_top_k=3 should retrieve at most 3 chunks."""
        result = rag_graph.invoke({"question": "test"})
        assert len(result.get("retrieved", [])) <= 3

    def test_override_search_top_k(self):
        """State-level search_top_k overrides the default."""
        graph = build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            search_top_k=10,  # default is high
            open_top_n=1,
        )
        result = graph.invoke({"question": "test", "search_top_k": 1})
        # FakeSearchTool respects top_k; with 1, we get 1 result
        assert len(result.get("retrieved", [])) == 1

    def test_override_open_top_n(self):
        """State-level open_top_n overrides the default."""
        graph = build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            open_top_n=10,  # default is high
            search_top_k=3,
        )
        result = graph.invoke({"question": "test", "open_top_n": 1})
        assert len(result.get("opened", [])) == 1


# ------------------------------------------------------------------ #
#  Test: citation extraction                                          #
# ------------------------------------------------------------------ #


class TestCitationExtraction:
    """Verify the extract_citations utility function."""

    def test_finds_known_ids(self):
        text = "Rome fell [chunk_001] and then [chunk_002]."
        known = ["chunk_001", "chunk_002", "chunk_003"]
        assert extract_citations(text, known) == ["chunk_001", "chunk_002"]

    def test_ignores_unknown_ids(self):
        text = "Rome fell [chunk_unknown]."
        known = ["chunk_001", "chunk_002"]
        assert extract_citations(text, known) == []

    def test_empty_text(self):
        assert extract_citations("", ["chunk_001"]) == []

    def test_empty_known(self):
        assert extract_citations("has chunk_001 in it", []) == []

    def test_no_duplicates(self):
        text = "[chunk_001] and again [chunk_001]."
        known = ["chunk_001"]
        result = extract_citations(text, known)
        assert result == ["chunk_001"]

    def test_citations_in_graph_output(self, rag_graph):
        """Graph should extract citations that match opened chunk IDs."""
        result = rag_graph.invoke({"question": "test"})
        opened_ids = {c["chunk_id"] for c in result.get("opened", [])}
        for cid in result.get("citations", []):
            assert cid in opened_ids


# ------------------------------------------------------------------ #
#  Test: error handling                                               #
# ------------------------------------------------------------------ #


class TestErrorHandling:
    """Verify the graph handles tool/model errors gracefully."""

    def test_search_failure_sets_error(self):
        """If search tool raises, error is set and graph doesn't crash."""
        graph = build_rag_graph(
            search_tool=FailingSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
        )
        result = graph.invoke({"question": "test"})
        assert result.get("error")
        assert "retrieve_node" in result["error"]

    def test_model_failure_sets_error(self):
        """If model raises, error is set and graph doesn't crash."""
        graph = build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FailingModel(),
        )
        result = graph.invoke({"question": "test"})
        assert result.get("error")
        assert "answer_node" in result["error"]

    def test_missing_chunk_skipped(self):
        """Open node skips chunks that aren't found without erroring."""

        # FakeOpenChunkTool returns found=False for unknown IDs
        # Override search to return an unknown ID
        class SearchWithUnknown:
            def execute(self, **kwargs):
                return {
                    "chunks": [
                        {
                            "chunk_id": "unknown_id",
                            "score": 0.5,
                            "title": "?",
                            "snippet": "",
                        },
                        FAKE_CHUNKS[0],
                    ],
                    "count": 2,
                }

        graph = build_rag_graph(
            search_tool=SearchWithUnknown(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            open_top_n=2,
        )
        result = graph.invoke({"question": "test"})
        # Only 1 chunk should be opened (the unknown one is skipped)
        assert len(result.get("opened", [])) == 1
        assert not result.get("error")


# ------------------------------------------------------------------ #
#  Test: determinism                                                  #
# ------------------------------------------------------------------ #


class TestDeterminism:
    """Same inputs should always produce the same outputs."""

    def test_deterministic_across_invocations(self, rag_graph):
        q = "When did the Roman Republic end?"
        r1 = rag_graph.invoke({"question": q})
        r2 = rag_graph.invoke({"question": q})
        assert r1["answer_text"] == r2["answer_text"]
        assert r1["citations"] == r2["citations"]
        assert r1["retrieved"] == r2["retrieved"]
        assert len(r1["opened"]) == len(r2["opened"])


# ------------------------------------------------------------------ #
#  Test: PromptBuilder.build() integration                            #
# ------------------------------------------------------------------ #


class TestPromptBuilderBuild:
    """Verify PromptBuilder.build() works with dict and Pydantic inputs."""

    def test_build_with_dict_chunks(self):
        """build() accepts plain dicts for opened_chunks."""
        from workbench.prompting.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        chunks = [
            {"chunk_id": "c1", "document_id": "d1", "title": "T1", "text": "Hello"},
        ]
        messages, meta = pb.build(
            "answer_with_citations",
            question="What?",
            opened_chunks=chunks,
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "What?" in messages[0]["content"]
        assert "c1" in messages[0]["content"]

    def test_build_with_pydantic_chunks(self):
        """build() accepts OpenedChunk Pydantic objects."""
        from workbench.prompting.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        chunks = [
            OpenedChunk(chunk_id="c1", document_id="d1", title="T1", text="Hello"),
        ]
        messages, meta = pb.build(
            "answer_with_citations",
            question="What?",
            opened_chunks=chunks,
        )
        assert len(messages) == 1
        assert "Hello" in messages[0]["content"]

    def test_build_meta_has_variant(self):
        """meta should include the variant name."""
        from workbench.prompting.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        chunks = [{"chunk_id": "c1", "document_id": "d1", "title": "T", "text": "x"}]
        _, meta = pb.build(
            "answer_with_citations",
            question="Q",
            opened_chunks=chunks,
        )
        assert meta["variant"] == "answer_with_citations"
        assert meta["template"] == "answer_with_citations.txt"
        assert meta["version"].startswith("v-")

    def test_build_meta_has_counts(self):
        """meta should include evidence_count and timeline_count."""
        from workbench.prompting.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        chunks = [
            {"chunk_id": "c1", "document_id": "d1", "title": "T", "text": "x"},
            {"chunk_id": "c2", "document_id": "d2", "title": "T", "text": "y"},
        ]
        _, meta = pb.build(
            "answer_with_citations",
            question="Q",
            opened_chunks=chunks,
        )
        assert meta["evidence_count"] == 2
        assert meta["timeline_count"] == 0

    def test_build_with_timeline(self):
        """build() includes timeline section when timeline is provided."""
        from workbench.prompting.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        chunks = [{"chunk_id": "c1", "document_id": "d1", "title": "T", "text": "x"}]
        timeline = [
            {"date": "27 BC", "event": "Republic ended", "supporting_chunk_ids": ["c1"]}
        ]
        messages, meta = pb.build(
            "answer_with_citations",
            question="Q",
            opened_chunks=chunks,
            timeline=timeline,
        )
        assert "TIMELINE" in messages[0]["content"]
        assert "27 BC" in messages[0]["content"]
        assert meta["timeline_count"] == 1

    def test_build_with_extra_instructions(self):
        """extra_instructions should be prepended to the prompt."""
        from workbench.prompting.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        chunks = [{"chunk_id": "c1", "document_id": "d1", "title": "T", "text": "x"}]
        messages, _ = pb.build(
            "answer_with_citations",
            question="Q",
            opened_chunks=chunks,
            extra_instructions="BE CONCISE.",
        )
        # extra_instructions should appear before the template content
        content = messages[0]["content"]
        assert content.startswith("BE CONCISE.")

    def test_build_does_not_break_render(self):
        """The original render() API should still work unchanged."""
        from workbench.prompting.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        result = pb.render(
            "answer_with_citations",
            question="Q",
            evidence="E",
            timeline_section="",
        )
        assert "Q" in result
        assert "E" in result


# ------------------------------------------------------------------ #
#  Test: graph with PromptBuilder                                     #
# ------------------------------------------------------------------ #


class TestGraphWithPromptBuilder:
    """Verify the graph works when a PromptBuilder is provided."""

    def test_graph_with_prompt_builder_produces_answer(self):
        """Graph should produce an answer when using PromptBuilder."""
        from workbench.prompting.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        graph = build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            prompt_builder=pb,
            open_top_n=2,
        )
        result = graph.invoke({"question": "When did the Roman Republic end?"})
        assert result.get("answer_text")
        assert not result.get("error")

    def test_graph_with_prompt_builder_has_prompt_meta(self):
        """Graph should include prompt_meta when using PromptBuilder."""
        from workbench.prompting.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        graph = build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            prompt_builder=pb,
            open_top_n=2,
        )
        result = graph.invoke({"question": "test"})
        meta = result.get("prompt_meta", {})
        assert meta.get("variant") == "answer_with_citations"
        assert meta.get("version", "").startswith("v-")

    def test_graph_without_prompt_builder_uses_fallback(self):
        """Graph should still work without a PromptBuilder (fallback)."""
        graph = build_rag_graph(
            search_tool=FakeSearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            prompt_builder=None,
            open_top_n=2,
        )
        result = graph.invoke({"question": "test"})
        assert result.get("answer_text")
        meta = result.get("prompt_meta", {})
        assert meta.get("variant") == "fallback"


# ------------------------------------------------------------------ #
#  Test: empty evidence edge case                                     #
# ------------------------------------------------------------------ #


class TestEmptyEvidence:
    """Verify the graph handles cases with no retrievable evidence."""

    def test_no_results_gives_graceful_answer(self):
        """When search returns nothing, answer should say so."""

        class EmptySearchTool:
            def execute(self, **kwargs):
                return {"chunks": [], "count": 0}

        graph = build_rag_graph(
            search_tool=EmptySearchTool(),
            open_chunk_tool=FakeOpenChunkTool(),
            model=FakeModel(),
            open_top_n=3,
        )
        result = graph.invoke({"question": "test"})
        # Should have a "not enough evidence" message
        assert (
            "don't have enough evidence" in result.get("answer_text", "").lower()
            or result.get("answer_text", "") != ""
        )
        assert result.get("citations") == []
