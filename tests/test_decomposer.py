"""
Tests for the task decomposer module.

Tests cover:
    1. Decomposition prompt parsing (SINGLE vs multi-part)
    2. Decompose node behaviour (enabled/disabled, error handling)
    3. Parallel retrieve node (fan-out, aggregation, error resilience)
    4. Synthesise node (merging sub-answers, citation preservation)
    5. Decompose dispatcher (conditional routing)
    6. End-to-end graph flow (decomposed vs simple path)

Run with::

    pytest tests/test_decomposer.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from workbench.graphs.decomposer import (
    MAX_SUB_QUESTIONS,
    _format_sub_answers,
    _parse_decompose_response,
    decompose_dispatcher,
    make_decompose_node,
    make_parallel_retrieve_node,
    make_synthesise_node,
)

# ------------------------------------------------------------------ #
#  Test fixtures / helpers                                            #
# ------------------------------------------------------------------ #


@dataclass
class FakeResponse:
    """Mimics the model.generate() return object."""

    text: str
    tokens_in: int = 10
    tokens_out: int = 5
    latency_ms: float = 50.0


class FakeModel:
    """A controllable fake language model for testing."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses) if responses else ["SINGLE"]
        self._call_count = 0

    def generate(self, messages: list[dict], **kwargs: Any) -> FakeResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return FakeResponse(text=self._responses[idx])


class FakeRagSubgraph:
    """
    A controllable fake rag_graph.

    Returns a canned result keyed by the question text.
    """

    def __init__(
        self,
        results: dict[str, dict[str, Any]] | None = None,
        default_result: dict[str, Any] | None = None,
    ):
        self._results = results or {}
        self._default = default_result or {
            "answer_text": "Default answer.",
            "citations": ["chunk_1"],
            "web_citations": [],
            "retrieved": [{"chunk_id": "chunk_1"}],
            "opened": [{"chunk_id": "chunk_1", "text": "Some evidence."}],
            "tokens_in": 100,
            "tokens_out": 50,
            "model_latency_ms": 200.0,
            "error": None,
        }

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        q = state.get("question", "")
        return self._results.get(q, self._default)


# Stub out observability so tests don't need a running tracer
@pytest.fixture(autouse=True)
def _mock_tracing():
    with patch("workbench.graphs.decomposer.start_span") as mock_span:
        mock_span.return_value.__enter__ = MagicMock(return_value=None)
        mock_span.return_value.__exit__ = MagicMock(return_value=False)
        with patch("workbench.graphs.decomposer.add_span_attributes"):
            yield


# ------------------------------------------------------------------ #
#  1. Parsing tests                                                   #
# ------------------------------------------------------------------ #


class TestParseDecomposeResponse:
    """Unit tests for _parse_decompose_response."""

    def test_single_uppercase(self):
        assert _parse_decompose_response("SINGLE") is None

    def test_single_lowercase(self):
        assert _parse_decompose_response("single") is None

    def test_single_with_whitespace(self):
        assert _parse_decompose_response("  SINGLE  \n") is None

    def test_single_with_explanation(self):
        """Model sometimes adds reasoning after SINGLE."""
        assert _parse_decompose_response("SINGLE - this is one question") is None

    def test_valid_json_array(self):
        result = _parse_decompose_response(
            '["When did WW1 start?", "When did WW1 end?"]'
        )
        assert result == ["When did WW1 start?", "When did WW1 end?"]

    def test_json_in_markdown_fences(self):
        text = '```json\n["Q1", "Q2"]\n```'
        result = _parse_decompose_response(text)
        assert result == ["Q1", "Q2"]

    def test_caps_at_max_sub_questions(self):
        many_qs = [f"Question {i}" for i in range(10)]
        text = str(many_qs).replace("'", '"')
        result = _parse_decompose_response(text)
        assert result is not None
        assert len(result) <= MAX_SUB_QUESTIONS

    def test_single_item_array_returns_none(self):
        """An array with just one element isn't a real decomposition."""
        result = _parse_decompose_response('["Only one question"]')
        assert result is None

    def test_empty_array_returns_none(self):
        result = _parse_decompose_response("[]")
        assert result is None

    def test_invalid_json_returns_none(self):
        result = _parse_decompose_response("not json at all {broken}")
        assert result is None

    def test_filters_empty_strings(self):
        result = _parse_decompose_response('["Q1", "", "Q2", "  "]')
        assert result == ["Q1", "Q2"]


# ------------------------------------------------------------------ #
#  2. Decompose node tests                                            #
# ------------------------------------------------------------------ #


class TestDecomposeNode:
    """Tests for make_decompose_node."""

    def test_disabled_returns_not_decomposed(self):
        node = make_decompose_node(model=FakeModel(), enabled=False)
        result = node({"question": "When did WW1 start and end?"})
        assert result["is_decomposed"] is False
        assert result["sub_questions"] == []

    def test_single_question_not_decomposed(self):
        model = FakeModel(responses=["SINGLE"])
        node = make_decompose_node(model=model, enabled=True)
        result = node({"question": "What caused WW1?"})
        assert result["is_decomposed"] is False

    def test_multi_part_question_decomposed(self):
        model = FakeModel(responses=['["When did WW1 start?", "When did WW1 end?"]'])
        node = make_decompose_node(model=model, enabled=True)
        result = node({"question": "When did WW1 start and end?"})
        assert result["is_decomposed"] is True
        assert len(result["sub_questions"]) == 2

    def test_uses_rewritten_query_when_available(self):
        """The node should prefer rewritten_query over raw question."""
        model = FakeModel(responses=["SINGLE"])
        node = make_decompose_node(model=model, enabled=True)

        state = {
            "question": "What about that?",
            "rewritten_query": "What about the Battle of Stalingrad?",
        }
        result = node(state)
        # The model was called — verify it didn't crash
        assert "is_decomposed" in result

    def test_model_error_returns_not_decomposed(self):
        """Decomposition failures are non-fatal."""
        model = MagicMock()
        model.generate.side_effect = RuntimeError("LLM down")
        node = make_decompose_node(model=model, enabled=True)
        result = node({"question": "When did WW1 start and end?"})
        assert result["is_decomposed"] is False
        assert result["sub_questions"] == []

    def test_records_latency(self):
        model = FakeModel(responses=["SINGLE"])
        node = make_decompose_node(model=model, enabled=True)
        result = node({"question": "Test?"})
        assert result["decompose_latency_ms"] >= 0

    def test_with_chat_history(self):
        """Chat context should be included without errors."""
        model = FakeModel(responses=["SINGLE"])
        node = make_decompose_node(model=model, enabled=True)
        result = node(
            {
                "question": "What about those events?",
                "chat_history": [
                    {"role": "user", "content": "Tell me about WW1"},
                    {"role": "assistant", "content": "WW1 was..."},
                ],
                "conversation_summary": "Discussing WW1.",
            }
        )
        assert result["is_decomposed"] is False


# ------------------------------------------------------------------ #
#  3. Parallel retrieve node tests                                    #
# ------------------------------------------------------------------ #


class TestParallelRetrieveNode:
    """Tests for make_parallel_retrieve_node."""

    def test_runs_all_sub_questions(self):
        rag = FakeRagSubgraph()
        node = make_parallel_retrieve_node(rag_subgraph=rag, max_workers=2)
        result = node(
            {
                "sub_questions": ["When did WW1 start?", "When did WW1 end?"],
            }
        )
        assert len(result["sub_results"]) == 2

    def test_aggregates_retrieved_and_opened(self):
        rag = FakeRagSubgraph(
            default_result={
                "answer_text": "Answer",
                "citations": ["c1"],
                "web_citations": [],
                "retrieved": [{"chunk_id": "c1"}],
                "opened": [{"chunk_id": "c1", "text": "text"}],
                "tokens_in": 50,
                "tokens_out": 25,
                "model_latency_ms": 100.0,
                "error": None,
            }
        )
        node = make_parallel_retrieve_node(rag_subgraph=rag, max_workers=2)
        result = node({"sub_questions": ["Q1", "Q2"]})

        # Two sub-questions, each returning 1 retrieved → 2 total
        assert len(result["retrieved"]) == 2
        assert len(result["opened"]) == 2
        assert result["tokens_in"] == 100  # 50 * 2
        assert result["tokens_out"] == 50  # 25 * 2

    def test_preserves_original_question_order(self):
        results_map = {
            "Q1": {
                "answer_text": "A1",
                "citations": [],
                "web_citations": [],
                "retrieved": [],
                "opened": [],
                "tokens_in": 0,
                "tokens_out": 0,
                "model_latency_ms": 0,
                "error": None,
            },
            "Q2": {
                "answer_text": "A2",
                "citations": [],
                "web_citations": [],
                "retrieved": [],
                "opened": [],
                "tokens_in": 0,
                "tokens_out": 0,
                "model_latency_ms": 0,
                "error": None,
            },
        }
        rag = FakeRagSubgraph(results=results_map)
        node = make_parallel_retrieve_node(rag_subgraph=rag, max_workers=2)
        result = node({"sub_questions": ["Q1", "Q2"]})
        questions = [sr["question"] for sr in result["sub_results"]]
        assert questions == ["Q1", "Q2"]

    def test_handles_subgraph_failure_gracefully(self):
        """One failing sub-question shouldn't break the others."""

        class FailingRag:
            def __init__(self):
                self._count = 0

            def invoke(self, state):
                self._count += 1
                if self._count == 1:
                    raise RuntimeError("Retrieval failed")
                return {
                    "answer_text": "Success",
                    "citations": [],
                    "web_citations": [],
                    "retrieved": [],
                    "opened": [],
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "model_latency_ms": 50.0,
                    "error": None,
                }

        node = make_parallel_retrieve_node(rag_subgraph=FailingRag(), max_workers=1)
        result = node({"sub_questions": ["Q1", "Q2"]})
        assert len(result["sub_results"]) == 2
        # One should have an error, one should have an answer
        errors = [sr for sr in result["sub_results"] if sr.get("error")]
        successes = [sr for sr in result["sub_results"] if sr.get("answer_text")]
        assert len(errors) == 1
        assert len(successes) == 1

    def test_passes_chat_context_to_subgraph(self):
        """Verify chat history is forwarded to each sub-invocation."""
        captured_states = []

        class CapturingRag:
            def invoke(self, state):
                captured_states.append(state)
                return {
                    "answer_text": "A",
                    "citations": [],
                    "web_citations": [],
                    "retrieved": [],
                    "opened": [],
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "model_latency_ms": 0,
                    "error": None,
                }

        node = make_parallel_retrieve_node(rag_subgraph=CapturingRag(), max_workers=1)
        node(
            {
                "sub_questions": ["Q1"],
                "chat_history": [{"role": "user", "content": "Hi"}],
                "conversation_summary": "Test summary",
            }
        )
        assert len(captured_states) == 1
        assert captured_states[0]["chat_history"] == [{"role": "user", "content": "Hi"}]
        assert captured_states[0]["conversation_summary"] == "Test summary"

    def test_deduplicates_citations(self):
        """Same citation from multiple sub-questions → appears once."""
        rag = FakeRagSubgraph(
            default_result={
                "answer_text": "Answer",
                "citations": ["shared_chunk"],
                "web_citations": [],
                "retrieved": [],
                "opened": [],
                "tokens_in": 0,
                "tokens_out": 0,
                "model_latency_ms": 0,
                "error": None,
            }
        )
        node = make_parallel_retrieve_node(rag_subgraph=rag, max_workers=2)
        result = node({"sub_questions": ["Q1", "Q2"]})
        assert result["citations"] == ["shared_chunk"]  # deduplicated


# ------------------------------------------------------------------ #
#  4. Synthesise node tests                                           #
# ------------------------------------------------------------------ #


class TestSynthesiseNode:
    """Tests for make_synthesise_node."""

    def test_combines_sub_answers(self):
        model = FakeModel(
            responses=["WW1 started in 1914 [chunk_a] and ended in 1918 [chunk_b]."]
        )
        node = make_synthesise_node(model=model)
        result = node(
            {
                "question": "When did WW1 start and end?",
                "sub_results": [
                    {
                        "question": "When did WW1 start?",
                        "answer_text": "WW1 started in 1914. [chunk_a]",
                        "citations": ["chunk_a"],
                        "web_citations": [],
                    },
                    {
                        "question": "When did WW1 end?",
                        "answer_text": "WW1 ended in 1918. [chunk_b]",
                        "citations": ["chunk_b"],
                        "web_citations": [],
                    },
                ],
                "tokens_in": 200,
                "tokens_out": 100,
            }
        )
        assert result["answer_text"]
        assert result["subgraph_used"] == "decomposer"

    def test_empty_sub_results_returns_no_evidence(self):
        model = FakeModel()
        node = make_synthesise_node(model=model)
        result = node({"question": "Test?", "sub_results": []})
        assert "don't have enough evidence" in result["answer_text"]

    def test_single_successful_result_passes_through(self):
        """If only one sub-answer succeeded, skip synthesis LLM call."""
        model = FakeModel(responses=["should not be called"])
        node = make_synthesise_node(model=model)
        result = node(
            {
                "question": "Q?",
                "sub_results": [
                    {
                        "question": "Q1",
                        "answer_text": "Direct answer. [chunk_1]",
                        "citations": ["chunk_1"],
                        "web_citations": [],
                    },
                    {
                        "question": "Q2",
                        "answer_text": "",  # failed
                        "citations": [],
                        "web_citations": [],
                    },
                ],
            }
        )
        assert result["answer_text"] == "Direct answer. [chunk_1]"
        assert result["prompt_meta"]["variant"] == "synthesise_single_passthrough"

    def test_synthesis_error_falls_back_to_concatenation(self):
        model = MagicMock()
        model.generate.side_effect = RuntimeError("LLM down")
        node = make_synthesise_node(model=model)
        result = node(
            {
                "question": "Q?",
                "sub_results": [
                    {"question": "Q1", "answer_text": "A1", "citations": []},
                    {"question": "Q2", "answer_text": "A2", "citations": []},
                ],
            }
        )
        assert "A1" in result["answer_text"]
        assert "A2" in result["answer_text"]
        assert result["error"]

    def test_accumulates_tokens(self):
        model = FakeModel(responses=["Combined answer."])
        node = make_synthesise_node(model=model)
        result = node(
            {
                "question": "Q?",
                "sub_results": [
                    {"question": "Q1", "answer_text": "A1", "citations": []},
                    {"question": "Q2", "answer_text": "A2", "citations": []},
                ],
                "tokens_in": 200,  # from parallel retrieve
                "tokens_out": 100,
            }
        )
        # Should include parallel tokens + synthesis tokens
        assert result["tokens_in"] == 200 + 10  # 10 from FakeResponse
        assert result["tokens_out"] == 100 + 5  # 5 from FakeResponse


# ------------------------------------------------------------------ #
#  5. Dispatcher tests                                                #
# ------------------------------------------------------------------ #


class TestDecomposeDispatcher:
    """Tests for decompose_dispatcher."""

    def test_decomposed_routes_to_parallel(self):
        state = {"is_decomposed": True, "sub_questions": ["Q1", "Q2"]}
        assert decompose_dispatcher(state) == "parallel_retrieve"

    def test_simple_routes_to_routing(self):
        state = {"is_decomposed": False, "sub_questions": []}
        assert decompose_dispatcher(state) == "route_question"

    def test_missing_fields_defaults_to_routing(self):
        assert decompose_dispatcher({}) == "route_question"

    def test_decomposed_but_empty_sub_questions_routes_to_routing(self):
        """Edge case: is_decomposed=True but no sub_questions."""
        state = {"is_decomposed": True, "sub_questions": []}
        assert decompose_dispatcher(state) == "route_question"


# ------------------------------------------------------------------ #
#  6. Format sub-answers helper                                       #
# ------------------------------------------------------------------ #


class TestFormatSubAnswers:
    """Tests for _format_sub_answers."""

    def test_formats_multiple_results(self):
        sub_results = [
            {"question": "Q1", "answer_text": "A1"},
            {"question": "Q2", "answer_text": "A2"},
        ]
        block = _format_sub_answers(sub_results)
        assert "[Part 1] Q1" in block
        assert "[Part 2] Q2" in block
        assert "A1" in block
        assert "A2" in block

    def test_handles_missing_answer(self):
        sub_results = [{"question": "Q1"}]
        block = _format_sub_answers(sub_results)
        assert "(no answer)" in block
