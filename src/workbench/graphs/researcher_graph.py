"""
Researcher graph: search → open → timeline → answer → validate (→ repair).

This graph extends the basic RAG graph with two additional capabilities:

1. **Timeline extraction** — calls the ``TimelineTool`` to build a
   chronological timeline from evidence *before* generating the answer.
   The timeline is included in the answer prompt so the model can
   write a better-structured, time-aware response.

2. **Citation validation + repair** — after the answer is generated,
   a validation node checks citation quality against a ``CitationPolicy``
   (default: ``RESEARCHER_STRICT``).  If validation fails *and*
   ``max_repair_attempts`` has not been exhausted, the graph routes
   to a repair node that re-prompts the model with explicit instructions
   to fix its citations, then re-validates.

Graph topology::

    retrieve → open → timeline → answer → validate ──[valid]──→ END
                                              ↑        │
                                              │   [invalid &
                                              │    retries left]
                                              │        │
                                              └─ repair ┘

Usage (standalone)::

    graph = build_researcher_graph(
        search_tool=search_tool,
        open_chunk_tool=open_chunk_tool,
        timeline_tool=timeline_tool,
        model=llm,
        prompt_builder=PromptBuilder(),
    )
    result = graph.invoke({"question": "When did the Roman Republic end?"})
    print(result["answer_text"])
    print(result["timeline"])

Usage (with observability)::

    result = run_researcher_graph(
        graph=graph,
        question="When did the Roman Republic end?",
        config_snapshot=snapshot,
    )
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, StateGraph

from workbench.agents.state import OpenedChunk, TimelineItem
from workbench.graphs.base_state import ResearcherState
from workbench.observability.tracing import add_span_attributes, start_span
from workbench.prompting.citations import (
    RESEARCHER_STRICT,
    CitationPolicy,
    ValidationResult,
    extract_citations,
    validate_citations,
)
from workbench.prompting.prompt_builder import (
    PromptBuilder,
    format_evidence_block,
    format_timeline_section,
)

# Re-export for convenience — rag_graph.extract_citations already
# existed; researcher_graph users can import from either place.
__all__ = [
    "build_researcher_graph",
    "run_researcher_graph",
]


# ------------------------------------------------------------------ #
#  Node factories                                                     #
# ------------------------------------------------------------------ #


def _make_retrieve_node(
    search_tool: Any,
    default_top_k: int,
) -> Any:
    """
    Create the retrieve node (identical to rag_graph's version).

    Calls ``search_tool.execute()`` with the question and writes
    the result into ``state["retrieved"]``.
    """

    def retrieve_node(state: ResearcherState) -> dict[str, Any]:
        question = state["question"]
        top_k = state.get("search_top_k", default_top_k)

        with start_span(
            "graph.retrieve_node",
            attributes={"question": question[:200], "top_k": top_k},
        ):
            try:
                result = search_tool.execute(query=question, top_k=top_k)
                chunks = result.get("chunks", [])
                add_span_attributes({"results_count": len(chunks)})
                return {"retrieved": chunks}

            except Exception as exc:
                return {
                    "retrieved": [],
                    "error": f"retrieve_node: {type(exc).__name__}: {exc}",
                }

    return retrieve_node


def _make_open_node(
    open_chunk_tool: Any,
    default_top_n: int,
) -> Any:
    """
    Create the open node (identical to rag_graph's version).

    Opens the top N retrieved chunks and writes full evidence
    into ``state["opened"]``.
    """

    def open_node(state: ResearcherState) -> dict[str, Any]:
        retrieved = state.get("retrieved", [])
        top_n = state.get("open_top_n", default_top_n)

        if state.get("error"):
            return {}

        with start_span(
            "graph.open_node",
            attributes={
                "retrieved_count": len(retrieved),
                "open_top_n": top_n,
            },
        ):
            opened: list[dict[str, Any]] = []
            for chunk_summary in retrieved[:top_n]:
                chunk_id = chunk_summary.get("chunk_id", "")
                if not chunk_id:
                    continue
                try:
                    result = open_chunk_tool.execute(chunk_id=chunk_id)
                    if result.get("found") and result.get("chunk"):
                        opened.append(result["chunk"])
                except Exception:
                    pass  # skip; don't abort the run

            add_span_attributes({"opened_count": len(opened)})
            return {"opened": opened}

    return open_node


def _make_timeline_node(
    timeline_tool: Any,
) -> Any:
    """
    Create the timeline node.

    Calls the ``TimelineTool`` with the opened evidence and writes
    a structured timeline (plus the raw model output) into state.
    """

    def timeline_node(state: ResearcherState) -> dict[str, Any]:
        opened_dicts = state.get("opened", [])
        question = state["question"]

        if state.get("error"):
            return {}

        if not opened_dicts:
            return {"timeline": [], "timeline_raw": ""}

        with start_span(
            "graph.timeline_node",
            attributes={"evidence_count": len(opened_dicts)},
        ):
            try:
                result = timeline_tool.execute(
                    question=question,
                    opened_chunks=opened_dicts,
                )

                timeline_items = result.get("timeline", [])
                raw_output = result.get("raw_model_output", "")

                add_span_attributes(
                    {
                        "timeline_count": len(timeline_items),
                        "cited_chunk_count": len(result.get("cited_chunk_ids", [])),
                    }
                )

                return {
                    "timeline": timeline_items,
                    "timeline_raw": raw_output,
                }

            except Exception as exc:
                # Timeline failure is non-fatal: continue without it.
                return {
                    "timeline": [],
                    "timeline_raw": "",
                    "error": (f"timeline_node: {type(exc).__name__}: {exc}"),
                }

    return timeline_node


def _make_answer_node(
    model: Any,
    prompt_builder: PromptBuilder | None,
    generation_settings: dict[str, Any] | None,
) -> Any:
    """
    Create the answer node.

    Same as rag_graph's answer node but always includes the
    timeline section in the prompt when available.
    """
    gen_settings = generation_settings or {}

    fallback_system = (
        "You are a meticulous historical researcher. Answer the user's "
        "question using ONLY the evidence provided below. For each claim, "
        "cite the source by writing [chunk_id] immediately after the claim. "
        "Every paragraph of your answer must contain at least one citation.\n\n"
        "A TIMELINE of events has been extracted from the evidence. Use it "
        "to structure your answer chronologically when appropriate.\n\n"
        "If the evidence does not contain enough information, say so honestly."
    )

    def answer_node(state: ResearcherState) -> dict[str, Any]:
        opened_dicts = state.get("opened", [])
        timeline_dicts = state.get("timeline", [])
        question = state["question"]

        # If upstream failed fatally, propagate
        if state.get("error") and not opened_dicts:
            return {}

        if not opened_dicts:
            return {
                "answer_text": (
                    "I don't have enough evidence in the provided "
                    "articles to fully answer this question."
                ),
                "citations": [],
                "tokens_in": 0,
                "tokens_out": 0,
                "model_latency_ms": 0.0,
                "prompt_meta": {},
            }

        with start_span(
            "graph.answer_node",
            attributes={
                "evidence_count": len(opened_dicts),
                "timeline_count": len(timeline_dicts),
            },
        ):
            try:
                # --- Build prompt ---------------------------------
                if prompt_builder is not None:
                    messages, meta = prompt_builder.build(
                        "answer_with_citations",
                        question=question,
                        opened_chunks=opened_dicts,
                        timeline=timeline_dicts,
                    )
                else:
                    # Fallback: inline prompt
                    opened_objs = [OpenedChunk(**d) for d in opened_dicts]
                    evidence_block = format_evidence_block(opened_objs)

                    tl_objs = [TimelineItem(**t) for t in timeline_dicts]
                    tl_section = format_timeline_section(tl_objs)

                    user_message = f"Evidence:\n{evidence_block}\n\n"
                    if tl_section:
                        user_message += f"{tl_section}\n\n"
                    user_message += (
                        f"Question: {question}\n\n"
                        "Answer the question using the evidence above. "
                        "Cite sources using [chunk_id] notation. "
                        "Every paragraph must have at least one citation."
                    )
                    messages = [
                        {"role": "system", "content": fallback_system},
                        {"role": "user", "content": user_message},
                    ]
                    meta = {
                        "variant": "fallback_researcher",
                        "template": "inline",
                    }

                # --- Call model -----------------------------------
                response = model.generate(messages, **gen_settings)

                # --- Extract citations ----------------------------
                known_ids = [d.get("chunk_id", "") for d in opened_dicts]
                citations = extract_citations(response.text, known_ids)

                add_span_attributes(
                    {
                        "answer_length": len(response.text),
                        "citation_count": len(citations),
                        "tokens_in": response.tokens_in,
                        "tokens_out": response.tokens_out,
                    }
                )

                return {
                    "answer_text": response.text,
                    "citations": citations,
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                    "model_latency_ms": response.latency_ms,
                    "prompt_meta": meta,
                }

            except Exception as exc:
                return {
                    "answer_text": "",
                    "citations": [],
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "model_latency_ms": 0.0,
                    "prompt_meta": {},
                    "error": f"answer_node: {type(exc).__name__}: {exc}",
                }

    return answer_node


def _make_validate_node(
    policy: CitationPolicy,
) -> Any:
    """
    Create the citation-validation node.

    Runs ``validate_citations()`` against the configured policy
    and writes the result into ``state["citation_validation"]``.
    The conditional edge after this node reads ``citation_validation``
    to decide whether to repair or finish.
    """

    def validate_node(state: ResearcherState) -> dict[str, Any]:
        answer_text = state.get("answer_text", "")
        opened_dicts = state.get("opened", [])
        timeline_dicts = state.get("timeline", [])
        opened_ids = [d.get("chunk_id", "") for d in opened_dicts]

        # Skip validation if we already have an error or empty answer
        if state.get("error") or not answer_text:
            return {
                "citation_validation": ValidationResult(valid=True).to_dict(),
            }

        with start_span(
            "graph.validate_node",
            attributes={
                "opened_count": len(opened_ids),
                "repair_attempts": state.get("repair_attempts", 0),
            },
        ):
            vr = validate_citations(
                answer_text=answer_text,
                opened_ids=opened_ids,
                policy=policy,
                timeline=timeline_dicts or None,
            )

            add_span_attributes(
                {
                    "validation_valid": vr.valid,
                    "validation_errors": len(vr.errors),
                    "validation_warnings": len(vr.warnings),
                    "cited_count": len(vr.cited_ids),
                    "hallucinated_count": len(vr.hallucinated_ids),
                }
            )

            return {"citation_validation": vr.to_dict()}

    return validate_node


def _make_repair_node(
    model: Any,
    prompt_builder: PromptBuilder | None,
    generation_settings: dict[str, Any] | None,
) -> Any:
    """
    Create the repair node.

    Re-prompts the model with the original evidence, the previous
    (failed) answer, and explicit instructions to fix citations.
    Increments ``repair_attempts``.
    """
    gen_settings = generation_settings or {}

    def repair_node(state: ResearcherState) -> dict[str, Any]:
        opened_dicts = state.get("opened", [])
        timeline_dicts = state.get("timeline", [])
        question = state["question"]
        previous_answer = state.get("answer_text", "")
        validation = state.get("citation_validation", {})
        repair_attempts = state.get("repair_attempts", 0)

        opened_ids = [d.get("chunk_id", "") for d in opened_dicts]

        with start_span(
            "graph.repair_node",
            attributes={"repair_attempt": repair_attempts + 1},
        ):
            try:
                # Build evidence + timeline context
                opened_objs = [OpenedChunk(**d) for d in opened_dicts]
                evidence_block = format_evidence_block(opened_objs)

                tl_objs = (
                    [TimelineItem(**t) for t in timeline_dicts]
                    if timeline_dicts
                    else []
                )
                tl_section = format_timeline_section(tl_objs)

                errors_text = "\n".join(f"- {e}" for e in validation.get("errors", []))

                repair_prompt = (
                    f"Your previous answer had citation problems:\n"
                    f"{errors_text}\n\n"
                    f"ALLOWED chunk IDs: {opened_ids}\n\n"
                    f"Evidence:\n{evidence_block}\n\n"
                )
                if tl_section:
                    repair_prompt += f"{tl_section}\n\n"
                repair_prompt += (
                    f"Question: {question}\n\n"
                    f"Your previous answer:\n{previous_answer}\n\n"
                    "Rewrite the answer. Rules:\n"
                    "1. Use ONLY chunk IDs from the ALLOWED set above.\n"
                    "2. Every paragraph must end with at least one "
                    "[chunk_id] citation.\n"
                    "3. Do NOT invent or hallucinate chunk IDs.\n"
                    "4. Keep the factual content; just fix the citations."
                )

                messages = [{"role": "user", "content": repair_prompt}]
                response = model.generate(messages, **gen_settings)

                citations = extract_citations(response.text, opened_ids)

                add_span_attributes(
                    {
                        "repair_answer_length": len(response.text),
                        "repair_citation_count": len(citations),
                    }
                )

                return {
                    "answer_text": response.text,
                    "citations": citations,
                    "tokens_in": (state.get("tokens_in", 0) + response.tokens_in),
                    "tokens_out": (state.get("tokens_out", 0) + response.tokens_out),
                    "model_latency_ms": (
                        state.get("model_latency_ms", 0.0) + response.latency_ms
                    ),
                    "repair_attempts": repair_attempts + 1,
                }

            except Exception as exc:
                return {
                    "repair_attempts": repair_attempts + 1,
                    "error": f"repair_node: {type(exc).__name__}: {exc}",
                }

    return repair_node


# ------------------------------------------------------------------ #
#  Routing                                                            #
# ------------------------------------------------------------------ #


def _make_should_repair(policy: CitationPolicy) -> Any:
    """
    Create the conditional routing function after validate_node.

    Returns ``"repair"`` if the validation failed and the repair
    budget has not been exhausted; otherwise returns ``"end"``.
    """

    def should_repair(state: ResearcherState) -> str:
        validation = state.get("citation_validation", {})
        repair_attempts = state.get("repair_attempts", 0)

        if (
            not validation.get("valid", True)
            and repair_attempts < policy.max_repair_attempts
        ):
            return "repair"
        return "end"

    return should_repair


# ------------------------------------------------------------------ #
#  Graph builder                                                      #
# ------------------------------------------------------------------ #


def build_researcher_graph(
    search_tool: Any,
    open_chunk_tool: Any,
    timeline_tool: Any,
    model: Any,
    prompt_builder: PromptBuilder | None = None,
    generation_settings: dict[str, Any] | None = None,
    open_top_n: int = 5,
    search_top_k: int = 10,
    citation_policy: CitationPolicy | None = None,
) -> Any:
    """
    Build and compile the researcher graph.

    Topology::

        retrieve → open → timeline → answer → validate ─→ END
                                                  ↑   │
                                                  │   └─ repair ─┘

    Args:
        search_tool:  SearchWikipediaTool or equivalent.
        open_chunk_tool:  OpenChunkTool or equivalent.
        timeline_tool:  TimelineTool or equivalent.
        model:  LanguageModel with ``.generate()``.
        prompt_builder:  Optional ``PromptBuilder`` instance.
        generation_settings:  Extra kwargs for ``model.generate()``.
        open_top_n:  Default chunks to open (overridable per-invoke).
        search_top_k:  Default chunks to retrieve (overridable).
        citation_policy:  Validation policy.  Defaults to
                          ``RESEARCHER_STRICT``.

    Returns:
        Compiled LangGraph ready for ``.invoke()``.
    """
    policy = citation_policy or RESEARCHER_STRICT

    builder: StateGraph = StateGraph(ResearcherState)

    # --- Register nodes -------------------------------------------
    builder.add_node(
        "retrieve",
        _make_retrieve_node(search_tool, default_top_k=search_top_k),
    )
    builder.add_node(
        "open",
        _make_open_node(open_chunk_tool, default_top_n=open_top_n),
    )
    builder.add_node(
        "timeline",
        _make_timeline_node(timeline_tool),
    )
    builder.add_node(
        "answer",
        _make_answer_node(model, prompt_builder, generation_settings),
    )
    builder.add_node(
        "validate",
        _make_validate_node(policy),
    )
    builder.add_node(
        "repair",
        _make_repair_node(model, prompt_builder, generation_settings),
    )

    # --- Wire edges -----------------------------------------------
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "open")
    builder.add_edge("open", "timeline")
    builder.add_edge("timeline", "answer")
    builder.add_edge("answer", "validate")

    # Conditional: repair or finish
    builder.add_conditional_edges(
        "validate",
        _make_should_repair(policy),
        {"repair": "repair", "end": END},
    )
    builder.add_edge("repair", "validate")

    return builder.compile()


# ------------------------------------------------------------------ #
#  Convenience runner (observability wrapper)                          #
# ------------------------------------------------------------------ #


def run_researcher_graph(
    graph: Any,
    question: str,
    config_snapshot: dict[str, Any] | None = None,
    *,
    search_top_k: int | None = None,
    open_top_n: int | None = None,
    run_type: str = "researcher_graph",
    service_name: str = "agentic-workbench-researcher-graph",
) -> ResearcherState:
    """
    Invoke the researcher graph with full observability setup.

    Sets up ``run_context``, tracing, logging, and metrics around
    a single graph invocation.

    Args:
        graph:  Compiled researcher graph.
        question:  The user question.
        config_snapshot:  Frozen config for run record.
        search_top_k:  Override default search_top_k.
        open_top_n:  Override default open_top_n.
        run_type:  Run type label.
        service_name:  Service name for tracing.

    Returns:
        Final ``ResearcherState`` dict.
    """
    from workbench.core.run_context import run_context
    from workbench.observability.logging import get_logger
    from workbench.observability.metrics import get_metrics_store
    from workbench.observability.tracing import setup_tracing

    snapshot = config_snapshot or {}
    setup_tracing(service_name=service_name)

    with run_context(snapshot, run_type=run_type) as ctx:
        logger = get_logger(run_id=ctx.run_id)
        metrics = get_metrics_store()

        logger.log_run_started(ctx.run_id, run_type, snapshot)
        logger.log_query(question)

        with start_span(
            "researcher_graph.run",
            attributes={"question": question[:200]},
        ):
            t0 = time.time()

            # --- Build initial state ------------------------------
            initial_state: dict[str, Any] = {"question": question}
            if search_top_k is not None:
                initial_state["search_top_k"] = search_top_k
            if open_top_n is not None:
                initial_state["open_top_n"] = open_top_n

            # --- Invoke -------------------------------------------
            result: ResearcherState = graph.invoke(initial_state)

            total_ms = (time.time() - t0) * 1000

        # --- Log results ------------------------------------------
        validation = result.get("citation_validation", {})
        logger.log_event(
            "researcher_graph_completed",
            {
                "question": question[:200],
                "answer_length": len(result.get("answer_text", "")),
                "citation_count": len(result.get("citations", [])),
                "retrieved_count": len(result.get("retrieved", [])),
                "opened_count": len(result.get("opened", [])),
                "timeline_count": len(result.get("timeline", [])),
                "tokens_in": result.get("tokens_in", 0),
                "tokens_out": result.get("tokens_out", 0),
                "repair_attempts": result.get("repair_attempts", 0),
                "validation_valid": validation.get("valid"),
                "validation_errors": validation.get("errors", []),
                "error": result.get("error"),
                "total_ms": round(total_ms, 1),
            },
        )

        if result.get("retrieved"):
            logger.log_retrieval(
                query=question,
                num_results=len(result["retrieved"]),
                retrieval_method="hybrid",
                duration_ms=total_ms,
            )

        tokens_out = result.get("tokens_out", 0)
        if tokens_out > 0:
            logger.log_model_call(
                model_name="local-model",
                tokens_in=result.get("tokens_in", 0),
                tokens_out=tokens_out,
                duration_ms=result.get("model_latency_ms", 0.0),
            )

        # --- Record metrics ---------------------------------------
        metrics.record_component_latency(
            component_name="researcher_graph.run",
            duration_ms=total_ms,
            component_type="graph",
        )

        if tokens_out > 0:
            metrics.record_model_tokens(
                model_name="local-model",
                tokens_in=result.get("tokens_in", 0),
                tokens_out=tokens_out,
                duration_ms=result.get("model_latency_ms", 0.0),
            )

        metrics.record_retrieval(
            final_count=len(result.get("opened", [])),
            query=question,
            keyword_candidates=len(result.get("retrieved", [])),
            vector_candidates=len(result.get("retrieved", [])),
            reranked=len(result.get("retrieved", [])),
            duration_ms=total_ms,
        )

        print(f"\nRun ID: {ctx.run_id}")
        return result
