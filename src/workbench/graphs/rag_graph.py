"""
RAG graph: search → open → answer as a compiled LangGraph.

This graph replaces the sequential ``AgentLoop`` + ``ScriptedPolicy``
combination with an explicit three-node DAG:

    retrieve_node  →  open_node  →  answer_node  →  END

Each node reads from ``RAGState``, calls the relevant tool or model,
and returns a partial state update dict.  The graph is compiled once
via ``build_rag_graph()`` and invoked per-question via ``invoke()``
or the convenience ``run_rag_graph()`` wrapper (which adds run_context,
logging, and metrics).

Usage (standalone)::

    graph = build_rag_graph(
        search_tool=search_tool,
        open_chunk_tool=open_chunk_tool,
        model=llm,
        prompt_builder=PromptBuilder(),
    )
    result = graph.invoke({"question": "When did the Roman Republic end?"})
    print(result["answer_text"])

Usage (with observability)::

    result = run_rag_graph(
        graph=graph,
        question="When did the Roman Republic end?",
        config_snapshot=snapshot,
    )
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, StateGraph

from workbench.agents.state import OpenedChunk
from workbench.graphs.base_state import RAGState
from workbench.observability.tracing import add_span_attributes, start_span
from workbench.prompting.prompt_builder import (
    PromptBuilder,
    format_evidence_block,
)

# ------------------------------------------------------------------ #
#  Utility                                                            #
# ------------------------------------------------------------------ #


def extract_citations(answer_text: str, known_ids: list[str]) -> list[str]:
    """
    Find chunk_ids mentioned in the answer text.

    Looks for patterns like [chunk_id] in the text.
    Only returns IDs that are in the known set.

    This is a standalone version of ``AgentLoop._extract_citations``
    so both the old loop and the new graph can share the logic.

    Args:
        answer_text: The generated answer string.
        known_ids: List of chunk IDs that were provided as evidence.

    Returns:
        List of cited chunk_ids (subset of known_ids found in the text).
    """
    cited: list[str] = []
    for cid in known_ids:
        if cid in answer_text:
            cited.append(cid)
    return cited


# ------------------------------------------------------------------ #
#  Node factories (closures over injected dependencies)               #
# ------------------------------------------------------------------ #


def _make_retrieve_node(
    search_tool: Any,
    default_top_k: int,
) -> Any:
    """
    Create the retrieve node function.

    The node calls ``search_tool.execute()`` with the question and
    writes the result into ``state["retrieved"]``.
    """

    def retrieve_node(state: RAGState) -> dict[str, Any]:
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
    Create the open node function.

    The node opens the top N retrieved chunks by calling
    ``open_chunk_tool.execute()`` for each one and writes the
    full evidence into ``state["opened"]``.
    """

    def open_node(state: RAGState) -> dict[str, Any]:
        retrieved = state.get("retrieved", [])
        top_n = state.get("open_top_n", default_top_n)

        # If retrieve failed, propagate
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
                    # Skip chunks that fail to open; don't abort the run
                    pass

            add_span_attributes({"opened_count": len(opened)})

            return {"opened": opened}

    return open_node


def _make_answer_node(
    model: Any,
    prompt_builder: PromptBuilder | None,
    generation_settings: dict[str, Any] | None,
) -> Any:
    """
    Create the answer node function.

    The node builds a prompt from the opened evidence, calls the
    language model, extracts citations, and writes the answer +
    metadata into state.
    """
    gen_settings = generation_settings or {}

    # Fallback system prompt — used when no PromptBuilder is provided.
    fallback_system = (
        "You are a helpful research assistant. Answer the user's question "
        "using ONLY the evidence provided below. For each claim, cite the "
        "source by writing [chunk_id] immediately after the claim.\n\n"
        "If the evidence does not contain enough information, say so honestly."
    )

    def answer_node(state: RAGState) -> dict[str, Any]:
        opened_dicts = state.get("opened", [])
        question = state["question"]

        # If upstream failed, propagate
        if state.get("error"):
            return {}

        # No evidence to answer from
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
            attributes={"evidence_count": len(opened_dicts)},
        ):
            try:
                # --- Build prompt ---------------------------------
                if prompt_builder is not None:
                    messages, meta = prompt_builder.build(
                        "answer_with_citations",
                        question=question,
                        opened_chunks=opened_dicts,
                    )
                else:
                    # Fallback: inline prompt (parity with old Day-8 loop)
                    opened_objs = [OpenedChunk(**d) for d in opened_dicts]
                    evidence_block = format_evidence_block(opened_objs)

                    user_message = (
                        f"Evidence:\n{evidence_block}\n\n"
                        f"Question: {question}\n\n"
                        "Answer the question using the evidence above. "
                        "Cite sources using [chunk_id] notation."
                    )
                    messages = [
                        {"role": "system", "content": fallback_system},
                        {"role": "user", "content": user_message},
                    ]
                    meta = {"variant": "fallback", "template": "inline"}

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


# ------------------------------------------------------------------ #
#  Graph builder                                                      #
# ------------------------------------------------------------------ #


def build_rag_graph(
    search_tool: Any,
    open_chunk_tool: Any,
    model: Any,
    prompt_builder: PromptBuilder | None = None,
    generation_settings: dict[str, Any] | None = None,
    open_top_n: int = 3,
    search_top_k: int = 5,
) -> Any:
    """
    Build and compile a three-node RAG graph.

    The graph replicates the behaviour of ``ScriptedPolicy`` +
    ``AgentLoop``:

        retrieve  →  open  →  answer  →  END

    All heavy dependencies (tools, model, prompt builder) are captured
    by closures so the compiled graph is a self-contained callable.

    Args:
        search_tool:  SearchWikipediaTool (or any ToolSpec with
                      ``.execute(query=..., top_k=...)``).
        open_chunk_tool:  OpenChunkTool (or any ToolSpec with
                          ``.execute(chunk_id=...)``).
        model:  LanguageModel with ``.generate(messages, **settings)``.
        prompt_builder:  Optional ``PromptBuilder`` instance.  When
                         provided, the answer node uses the
                         ``answer_with_citations`` template.  When
                         ``None``, a built-in fallback prompt is used.
        generation_settings:  Extra kwargs forwarded to ``model.generate``
                              (e.g. temperature, max_tokens).
        open_top_n:  Default number of retrieved chunks to open.
                     Can be overridden per-invoke via state.
        search_top_k:  Default number of chunks to retrieve.
                       Can be overridden per-invoke via state.

    Returns:
        A compiled LangGraph (``CompiledGraph``) ready for ``.invoke()``.
    """
    builder: StateGraph = StateGraph(RAGState)

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
        "answer",
        _make_answer_node(model, prompt_builder, generation_settings),
    )

    # --- Wire edges -----------------------------------------------
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "open")
    builder.add_edge("open", "answer")
    builder.add_edge("answer", END)

    return builder.compile()


# ------------------------------------------------------------------ #
#  Convenience runner (observability wrapper)                          #
# ------------------------------------------------------------------ #


def run_rag_graph(
    graph: Any,
    question: str,
    config_snapshot: dict[str, Any] | None = None,
    *,
    search_top_k: int | None = None,
    open_top_n: int | None = None,
    run_type: str = "rag_graph",
    service_name: str = "agentic-workbench-rag-graph",
) -> RAGState:
    """
    Invoke the RAG graph with full observability setup.

    Sets up ``run_context``, tracing, logging, and metrics around
    a single graph invocation — the "graph runner" pattern.

    Args:
        graph:  Compiled RAG graph from ``build_rag_graph()``.
        question:  The user question.
        config_snapshot:  Frozen config dict for the run record.
                          Pass ``{}`` if you don't need config tracking.
        search_top_k:  Override the default search_top_k.
        open_top_n:  Override the default open_top_n.
        run_type:  Run type label for the context.
        service_name:  Service name for tracing.

    Returns:
        Final ``RAGState`` dict with answer, citations, and metadata.
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
            "rag_graph.run",
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
            result: RAGState = graph.invoke(initial_state)

            total_ms = (time.time() - t0) * 1000

        # --- Log results ------------------------------------------
        logger.log_event(
            "rag_graph_completed",
            {
                "question": question[:200],
                "answer_length": len(result.get("answer_text", "")),
                "citation_count": len(result.get("citations", [])),
                "retrieved_count": len(result.get("retrieved", [])),
                "opened_count": len(result.get("opened", [])),
                "tokens_in": result.get("tokens_in", 0),
                "tokens_out": result.get("tokens_out", 0),
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
            component_name="rag_graph.run",
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
