"""
Task decomposer: splits multi-part questions into parallel sub-retrievals.

When a user asks a complex question with multiple independent parts
(e.g., "When did WW1 start and when did it end?"), this module:

1. Uses the LLM to detect multi-part questions and extract sub-questions
2. Runs each sub-question through the RAG pipeline in parallel
3. Synthesises all sub-answers into a single coherent response

Integration
-----------
The three node factories (``_make_decompose_node``,
``_make_parallel_retrieve_node``, ``_make_synthesise_node``) are
imported by ``supervisor_graph.py`` and wired as a pre-routing path.
Simple questions pass through unchanged to normal routing.

Supervisor topology after integration::

    decompose_check ──[needs_decomposition]──→ parallel_retrieve → synthesise → merge → END
                    ──[simple]──────────────→ route_question → [dispatch] → … → merge → END

Usage (standalone, for testing)::

    from workbench.graphs.decomposer import (
        make_decompose_node,
        make_parallel_retrieve_node,
        make_synthesise_node,
    )

Dependencies
------------
- ``concurrent.futures`` for parallel sub-graph invocations
- Existing ``rag_graph`` compiled subgraph for per-question retrieval
- LLM via ``model.generate(messages, **settings)``
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from workbench.observability.tracing import add_span_attributes, start_span
from workbench.prompting.prompt_builder import (
    PromptBuilder,
    format_conversation_context,
)

# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

# Maximum sub-questions we'll allow (safety cap)
MAX_SUB_QUESTIONS = 5

# Default parallelism for sub-graph invocations
DEFAULT_MAX_WORKERS = 4

# ------------------------------------------------------------------ #
#  Decomposition prompt (fallback when no PromptBuilder template)     #
# ------------------------------------------------------------------ #

_FALLBACK_DECOMPOSE_PROMPT = """\
You are a question analyser.  Determine whether the following question \
contains multiple independent sub-questions that would each benefit from \
a separate search of the knowledge base.

RULES:
- "When did WW1 start and when did it end?" → TWO sub-questions \
(start date, end date)
- "What caused WW1?" → SINGLE (complex answer but one research topic)
- "Who was Napoleon and what battles did he fight?" → TWO sub-questions
- "What is the capital of France?" → SINGLE
- Only split when parts are truly independent retrieval tasks.
- Each sub-question must be fully self-contained (include enough \
context from the original so it can be searched on its own).

{conversation_context}
QUESTION: {question}

If the question is simple (one retrieval task), respond with exactly:
SINGLE

If the question has multiple independent parts, respond with ONLY a \
JSON array of self-contained sub-questions — no commentary:
["sub-question 1", "sub-question 2"]\
"""

# ------------------------------------------------------------------ #
#  Synthesis prompt (fallback when no PromptBuilder template)         #
# ------------------------------------------------------------------ #

_FALLBACK_SYNTHESISE_PROMPT = """\
You are a research assistant.  The user asked a multi-part question.  \
Each part was researched separately.  Your job is to combine the \
sub-answers into a single coherent response.

RULES:
1. Preserve ALL citations exactly as they appear (e.g. [chunk_id]).
2. Write a unified, paragraph-based answer — not a list of separate answers.
3. If a sub-answer says it lacks evidence, acknowledge that gap.
4. Keep the answer concise and well-structured.

{conversation_context}
ORIGINAL QUESTION: {question}

--- SUB-ANSWERS ---
{sub_answers_block}

Write your combined answer now, preserving all citations.\
"""

# ------------------------------------------------------------------ #
#  Parsing helpers                                                    #
# ------------------------------------------------------------------ #


def _parse_decompose_response(text: str) -> list[str] | None:
    """
    Parse the LLM's decomposition response.

    Returns:
        ``None`` if the model said SINGLE (no decomposition needed).
        A list of sub-question strings if decomposition was requested.
    """
    cleaned = text.strip()

    # Check for SINGLE verdict
    if cleaned.upper().startswith("SINGLE"):
        return None

    # Try to extract a JSON array
    # The model might wrap it in markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
            # Safety: cap and filter empties
            sub_qs = [q.strip() for q in parsed if q.strip()]
            if len(sub_qs) >= 2:
                return sub_qs[:MAX_SUB_QUESTIONS]
    except (json.JSONDecodeError, TypeError):
        pass

    # If we can't parse it, treat as SINGLE (safe fallback)
    return None


def _format_sub_answers(sub_results: list[dict[str, Any]]) -> str:
    """Format sub-question results into a text block for the synthesis prompt."""
    parts: list[str] = []
    for i, sr in enumerate(sub_results, 1):
        q = sr.get("question", f"Sub-question {i}")
        a = sr.get("answer_text", "(no answer)")
        parts.append(f"[Part {i}] {q}\n{a}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------ #
#  Node factories                                                     #
# ------------------------------------------------------------------ #


def make_decompose_node(
    model: Any,
    prompt_builder: PromptBuilder | None = None,
    generation_settings: dict[str, Any] | None = None,
    enabled: bool = True,
) -> Any:
    """
    Create the decomposition-check node.

    When ``enabled=False``, the node always returns ``is_decomposed=False``
    and the supervisor falls through to normal routing.

    When enabled, calls the LLM to decide whether the question has
    multiple independent parts.  If yes, extracts sub-questions.

    State reads:
        ``question``, ``chat_history``, ``conversation_summary``,
        ``rewritten_query``

    State writes:
        ``is_decomposed``, ``sub_questions``, ``decompose_latency_ms``
    """
    gen_settings = {
        **(generation_settings or {}),
        "temperature": 0.1,
        "max_tokens": 300,
    }

    def decompose_node(state: dict[str, Any]) -> dict[str, Any]:
        # Use the rewritten query if available (chat coreference resolved),
        # otherwise fall back to the raw question
        question = state.get("rewritten_query") or state["question"]
        chat_history = state.get("chat_history", [])
        conversation_summary = state.get("conversation_summary", "")

        # --- Disabled: pass through immediately -----------------------
        if not enabled:
            return {
                "is_decomposed": False,
                "sub_questions": [],
                "decompose_latency_ms": 0.0,
            }

        with start_span(
            "graph.decompose_check",
            attributes={"question": question[:200], "enabled": enabled},
        ):
            t0 = time.time()

            try:
                # Build conversation context for chat-aware decomposition
                conversation_context = ""
                if chat_history:
                    conversation_context = (
                        "CONVERSATION CONTEXT:\n"
                        + format_conversation_context(
                            chat_history=chat_history,
                            conversation_summary=conversation_summary,
                        )
                        + "\n"
                    )

                # Build prompt
                if prompt_builder is not None:
                    try:
                        user_prompt = prompt_builder.render(
                            "decompose_question",
                            question=question,
                            conversation_context=conversation_context,
                        )
                    except FileNotFoundError:
                        user_prompt = _FALLBACK_DECOMPOSE_PROMPT.format(
                            question=question,
                            conversation_context=conversation_context,
                        )
                else:
                    user_prompt = _FALLBACK_DECOMPOSE_PROMPT.format(
                        question=question,
                        conversation_context=conversation_context,
                    )

                messages = [{"role": "user", "content": user_prompt}]
                response = model.generate(messages, **gen_settings)

                sub_questions = _parse_decompose_response(response.text)
                is_decomposed = sub_questions is not None

                latency_ms = (time.time() - t0) * 1000

                add_span_attributes(
                    {
                        "is_decomposed": is_decomposed,
                        "sub_question_count": len(sub_questions)
                        if sub_questions
                        else 0,
                        "decompose_latency_ms": latency_ms,
                        "raw_response": response.text[:300],
                    }
                )

                return {
                    "is_decomposed": is_decomposed,
                    "sub_questions": sub_questions or [],
                    "decompose_latency_ms": latency_ms,
                }

            except Exception as exc:
                # Decomposition failure is non-fatal — treat as SINGLE
                latency_ms = (time.time() - t0) * 1000
                add_span_attributes({"decompose_error": str(exc)[:200]})
                return {
                    "is_decomposed": False,
                    "sub_questions": [],
                    "decompose_latency_ms": latency_ms,
                }

    return decompose_node


def decompose_dispatcher(state: dict[str, Any]) -> str:
    """
    Conditional edge: route to parallel retrieval or normal routing.

    Returns:
        ``"parallel_retrieve"`` if the question was decomposed.
        ``"route_question"`` for simple (non-decomposed) questions.
    """
    if state.get("is_decomposed", False) and state.get("sub_questions"):
        return "parallel_retrieve"
    return "route_question"


def make_parallel_retrieve_node(
    rag_subgraph: Any,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Any:
    """
    Create the parallel retrieval node.

    Invokes the compiled ``rag_subgraph`` once per sub-question,
    concurrently via ``ThreadPoolExecutor``.  Collects all results
    into ``state["sub_results"]``.

    Each sub-invocation gets the full chat context so that
    ``rewrite_query`` inside the rag_graph can resolve coreferences.

    State reads:
        ``sub_questions``, ``chat_history``, ``conversation_summary``

    State writes:
        ``sub_results``, ``retrieved``, ``opened``, ``citations``,
        ``tokens_in``, ``tokens_out``
    """

    def parallel_retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
        sub_questions = state.get("sub_questions", [])
        chat_history = state.get("chat_history", [])
        conversation_summary = state.get("conversation_summary", "")

        with start_span(
            "graph.parallel_retrieve",
            attributes={
                "sub_question_count": len(sub_questions),
                "max_workers": max_workers,
            },
        ):
            t0 = time.time()

            def _run_sub(question: str) -> dict[str, Any]:
                """Invoke rag_subgraph for a single sub-question."""
                sub_input: dict[str, Any] = {"question": question}
                if chat_history:
                    sub_input["chat_history"] = chat_history
                if conversation_summary:
                    sub_input["conversation_summary"] = conversation_summary

                result = rag_subgraph.invoke(sub_input)

                # Use `or []` (not just `.get(k, [])`) because
                # LangGraph may set list fields to None rather
                # than leaving them absent from the state dict.
                return {
                    "question": question,
                    "answer_text": result.get("answer_text") or "",
                    "citations": result.get("citations") or [],
                    "web_citations": result.get("web_citations") or [],
                    "retrieved": result.get("retrieved") or [],
                    "opened": result.get("opened") or [],
                    "tokens_in": result.get("tokens_in") or 0,
                    "tokens_out": result.get("tokens_out") or 0,
                    "model_latency_ms": result.get("model_latency_ms") or 0.0,
                    "error": result.get("error"),
                }

            # --- Fan-out: run all sub-questions in parallel ------------
            sub_results: list[dict[str, Any]] = []

            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_question = {
                        executor.submit(_run_sub, q): q for q in sub_questions
                    }

                    for future in as_completed(future_to_question):
                        question = future_to_question[future]
                        try:
                            result = future.result()
                            sub_results.append(result)
                        except Exception as exc:
                            sub_results.append(
                                {
                                    "question": question,
                                    "answer_text": "",
                                    "citations": [],
                                    "web_citations": [],
                                    "retrieved": [],
                                    "opened": [],
                                    "tokens_in": 0,
                                    "tokens_out": 0,
                                    "model_latency_ms": 0.0,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
            except Exception as exc:
                # ThreadPool itself failed — still return what we have
                add_span_attributes({"parallel_error": str(exc)[:200]})

            # --- Sort sub_results to match original question order -----
            order = {q: i for i, q in enumerate(sub_questions)}
            sub_results.sort(key=lambda r: order.get(r.get("question", ""), 999))

            # --- Aggregate metadata across sub-results -----------------
            all_retrieved: list[dict[str, Any]] = []
            all_opened: list[dict[str, Any]] = []
            all_citations: list[str] = []
            all_web_citations: list[str] = []
            total_tokens_in = 0
            total_tokens_out = 0

            for sr in sub_results:
                all_retrieved.extend(sr.get("retrieved") or [])
                all_opened.extend(sr.get("opened") or [])
                all_citations.extend(sr.get("citations") or [])
                all_web_citations.extend(sr.get("web_citations") or [])
                total_tokens_in += sr.get("tokens_in") or 0
                total_tokens_out += sr.get("tokens_out") or 0

            latency_ms = (time.time() - t0) * 1000

            add_span_attributes(
                {
                    "sub_results_count": len(sub_results),
                    "total_retrieved": len(all_retrieved),
                    "total_opened": len(all_opened),
                    "parallel_latency_ms": latency_ms,
                }
            )

            return {
                "sub_results": sub_results,
                "retrieved": all_retrieved,
                "opened": all_opened,
                "citations": list(dict.fromkeys(all_citations)),  # dedupe, keep order
                "web_citations": list(dict.fromkeys(all_web_citations)),
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
            }

    return parallel_retrieve_node


def make_synthesise_node(
    model: Any,
    prompt_builder: PromptBuilder | None = None,
    generation_settings: dict[str, Any] | None = None,
) -> Any:
    """
    Create the synthesis node.

    Takes the sub-results produced by ``parallel_retrieve_node``
    and calls the LLM to combine them into a single coherent answer
    that preserves all citations.

    State reads:
        ``question``, ``sub_results``, ``chat_history``,
        ``conversation_summary``

    State writes:
        ``answer_text``, ``citations``, ``web_citations``,
        ``tokens_in``, ``tokens_out``, ``model_latency_ms``,
        ``prompt_meta``, ``subgraph_used``
    """
    gen_settings = generation_settings or {}

    def synthesise_node(state: dict[str, Any]) -> dict[str, Any]:
        question = state["question"]
        sub_results = state.get("sub_results", [])
        chat_history = state.get("chat_history", [])
        conversation_summary = state.get("conversation_summary", "")

        # Edge case: no sub-results at all
        if not sub_results:
            return {
                "answer_text": (
                    "I don't have enough evidence in the provided "
                    "articles to fully answer this question."
                ),
                "subgraph_used": "decomposer",
                "prompt_meta": {"variant": "synthesise_empty"},
            }

        # If only one sub-result came back (e.g. one sub-q failed),
        # just return that answer directly — no synthesis needed
        successful = [sr for sr in sub_results if sr.get("answer_text")]
        if len(successful) == 1:
            sr = successful[0]
            return {
                "answer_text": sr["answer_text"],
                "citations": sr.get("citations", []),
                "web_citations": sr.get("web_citations", []),
                "tokens_in": state.get("tokens_in", 0),
                "tokens_out": state.get("tokens_out", 0),
                "model_latency_ms": sr.get("model_latency_ms", 0.0),
                "prompt_meta": {"variant": "synthesise_single_passthrough"},
                "subgraph_used": "decomposer",
            }

        with start_span(
            "graph.synthesise",
            attributes={
                "question": question[:200],
                "sub_results_count": len(sub_results),
            },
        ):
            try:
                # Build conversation context
                conversation_context = ""
                if chat_history:
                    conversation_context = (
                        "CONVERSATION CONTEXT:\n"
                        + format_conversation_context(
                            chat_history=chat_history,
                            conversation_summary=conversation_summary,
                        )
                        + "\n"
                    )

                sub_answers_block = _format_sub_answers(sub_results)

                # Build prompt
                if prompt_builder is not None:
                    try:
                        user_prompt = prompt_builder.render(
                            "synthesize_answers",
                            question=question,
                            conversation_context=conversation_context,
                            sub_answers_block=sub_answers_block,
                        )
                    except FileNotFoundError:
                        user_prompt = _FALLBACK_SYNTHESISE_PROMPT.format(
                            question=question,
                            conversation_context=conversation_context,
                            sub_answers_block=sub_answers_block,
                        )
                else:
                    user_prompt = _FALLBACK_SYNTHESISE_PROMPT.format(
                        question=question,
                        conversation_context=conversation_context,
                        sub_answers_block=sub_answers_block,
                    )

                messages = [{"role": "user", "content": user_prompt}]
                response = model.generate(messages, **gen_settings)

                # Aggregate tokens: sub-retrieval tokens + synthesis tokens
                prev_tokens_in = state.get("tokens_in", 0)
                prev_tokens_out = state.get("tokens_out", 0)

                # Re-extract citations from the synthesised answer
                # (the LLM should preserve them, but let's verify)
                all_known_ids: list[str] = []
                all_known_urls: list[str] = []
                for sr in sub_results:
                    all_known_ids.extend(sr.get("citations") or [])
                    all_known_urls.extend(sr.get("web_citations") or [])

                from workbench.graphs.rag_graph import (
                    extract_citations,
                    extract_web_citations,
                )

                final_citations = extract_citations(
                    response.text, list(dict.fromkeys(all_known_ids))
                )
                final_web_citations = extract_web_citations(
                    response.text, list(dict.fromkeys(all_known_urls))
                )

                add_span_attributes(
                    {
                        "synthesised_answer_length": len(response.text),
                        "final_citation_count": len(final_citations),
                        "synthesis_tokens_in": response.tokens_in,
                        "synthesis_tokens_out": response.tokens_out,
                    }
                )

                return {
                    "answer_text": response.text,
                    "citations": final_citations,
                    "web_citations": final_web_citations,
                    "tokens_in": prev_tokens_in + response.tokens_in,
                    "tokens_out": prev_tokens_out + response.tokens_out,
                    "model_latency_ms": response.latency_ms,
                    "prompt_meta": {
                        "variant": "synthesise_decomposed",
                        "sub_question_count": len(sub_results),
                    },
                    "subgraph_used": "decomposer",
                }

            except Exception as exc:
                # Synthesis failed — concatenate sub-answers as fallback
                fallback_parts = []
                for sr in sub_results:
                    if sr.get("answer_text"):
                        fallback_parts.append(sr["answer_text"])

                return {
                    "answer_text": "\n\n".join(fallback_parts)
                    if fallback_parts
                    else "",
                    "subgraph_used": "decomposer",
                    "prompt_meta": {"variant": "synthesise_fallback"},
                    "error": f"synthesise_node: {type(exc).__name__}: {exc}",
                }

    return synthesise_node
