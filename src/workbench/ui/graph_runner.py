"""
Graph execution with streaming UI updates and citation post-processing.

This module runs a compiled LangGraph with ``.stream()`` to show
live node progress in Streamlit, then applies ``numberize_result()``
to replace raw ``[chunk_id_hash]`` references with clean ``[1]``,
``[2]``, etc.

This is the single integration point for the numberization feature —
it works identically for all three graph types (rag, researcher,
supervisor).
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from workbench.ui.chat_state import build_chat_state

# ------------------------------------------------------------------ #
#  Human-readable labels for graph node events                         #
# ------------------------------------------------------------------ #

NODE_LABELS: dict[str, str] = {
    # RAG / researcher nodes
    "rewrite_query": "🔄 Rewriting query for context…",
    "retrieve": "🔍 Searching knowledge base…",
    "check_confidence": "📊 Checking retrieval confidence…",
    "web_search": "🌐 Running web search fallback…",
    "open": "📖 Opening top chunks…",
    "timeline": "📅 Extracting timeline…",
    "answer": "✍️ Generating answer…",
    "validate": "✅ Validating citations…",
    "repair": "🔧 Repairing citations…",
    # Supervisor nodes
    "route_question": "🧭 Routing question to specialist…",
    "run_local_rag": "📚 Running local RAG pipeline…",
    "run_web_research": "🌐 Running web research…",
    "run_timeline_research": "📅 Running timeline researcher…",
    "run_math": "🔢 Evaluating math expression…",
    "merge": "🔀 Merging results…",
}


# ------------------------------------------------------------------ #
#  Streaming graph runner                                              #
# ------------------------------------------------------------------ #


def run_graph_streaming(
    graph,
    question: str,
    graph_name: str,
    cfg,
) -> dict[str, Any]:
    """
    Run a compiled LangGraph with .stream() and show node progress.

    After the graph finishes, applies ``numberize_result()`` to replace
    raw chunk ID hashes in the answer with sequential ``[1]``, ``[2]``
    references and build a ``citation_map`` for the UI renderers.

    Injects chat history from session state into the initial graph state
    so the rewrite_query node can resolve coreferences.

    Returns the final accumulated state dict (with numberized citations).
    """
    from workbench.core.config import create_config_snapshot
    from workbench.core.run_context import run_context
    from workbench.observability.logging import get_logger
    from workbench.observability.metrics import get_metrics_store
    from workbench.observability.tracing import setup_tracing, start_span
    from workbench.prompting.citations import numberize_result

    snapshot = create_config_snapshot(cfg)
    setup_tracing(service_name=f"agentic-workbench-{graph_name}")

    with run_context(snapshot, run_type=graph_name) as ctx:
        run_id = ctx.run_id
        logger = get_logger(run_id=run_id)
        metrics = get_metrics_store()

        logger.log_run_started(run_id, graph_name, snapshot)
        logger.log_query(question)

        # --- Build initial state with chat history ----------------
        initial_state: dict[str, Any] = {"question": question}

        chat_state = build_chat_state()
        if chat_state["chat_history"]:
            initial_state["chat_history"] = chat_state["chat_history"]
        if chat_state["conversation_summary"]:
            initial_state["conversation_summary"] = chat_state["conversation_summary"]

        # --- Stream through graph nodes with live UI updates ------
        status_container = st.status(
            f"Running **{graph_name}**…",
            expanded=True,
        )

        accumulated: dict[str, Any] = dict(initial_state)
        t0 = time.time()

        with status_container:
            try:
                with start_span(
                    f"{graph_name}.run",
                    attributes={"question": question[:200]},
                ):
                    for event in graph.stream(initial_state):
                        for node_name, update in event.items():
                            label = NODE_LABELS.get(
                                node_name,
                                f"⚙️ Running {node_name}…",
                            )
                            st.write(label)

                            _render_node_caption(node_name, update, question)

                            accumulated.update(update)

            except Exception as exc:
                st.error(f"Graph execution failed: {type(exc).__name__}: {exc}")
                accumulated["error"] = str(exc)

        total_ms = (time.time() - t0) * 1000

        # Update status label
        if accumulated.get("error"):
            status_container.update(
                label=f"**{graph_name}** finished with error",
                state="error",
                expanded=False,
            )
        else:
            status_container.update(
                label=f"**{graph_name}** completed in {total_ms:.0f} ms",
                state="complete",
                expanded=False,
            )

        # --- Post-process: numberize citations --------------------
        numberize_result(accumulated)

        # --- Logging / metrics ------------------------------------
        _log_run(
            logger,
            metrics,
            graph_name,
            question,
            accumulated,
            chat_state,
            total_ms,
            cfg,
        )

        accumulated["_run_id"] = run_id

        return accumulated


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #


def _render_node_caption(node_name: str, update: dict[str, Any], question: str) -> None:
    """Show brief stats / context for a streaming node's output."""
    if node_name == "rewrite_query" and "rewritten_query" in update:
        rq = update["rewritten_query"]
        if rq != question:
            st.caption(f"  → Rewritten: *{rq}*")
        else:
            st.caption("  → No rewrite needed")

    elif node_name == "retrieve" and "retrieved" in update:
        count = len(update["retrieved"])
        st.caption(f"  → Found {count} candidate chunks")

    elif node_name == "open" and "opened" in update:
        count = len(update["opened"])
        titles = [d.get("title", "?") for d in update["opened"][:5]]
        st.caption(f"  → Opened {count} chunks: {', '.join(titles)}")

    elif node_name == "timeline" and "timeline" in update:
        count = len(update["timeline"])
        st.caption(f"  → Extracted {count} timeline events")

    elif node_name == "web_search" and "web_results" in update:
        count = len(update["web_results"])
        st.caption(f"  → Got {count} web results")

    elif node_name == "validate" and "citation_validation" in update:
        valid = update["citation_validation"].get("valid", "?")
        st.caption(f"  → Validation passed: {valid}")

    elif node_name == "repair":
        st.caption("  → Re-generating with citation fixes…")

    elif node_name == "route_question" and "route" in update:
        route = update["route"]
        method = update.get("router_method", "?")
        conf = update.get("router_confidence", 0)
        st.caption(f"  → Route: **{route}** (method={method}, conf={conf:.0%})")

    elif node_name == "run_math" and "calculator_result" in update:
        calc = update["calculator_result"]
        if calc and calc.get("result") is not None:
            st.caption(f"  → {calc['expression']} = {calc['result']}")

    elif node_name == "run_web_research":
        wr = update.get("web_results", [])
        st.caption(f"  → Got {len(wr)} web results")

    elif node_name == "run_timeline_research":
        tl = update.get("timeline", [])
        st.caption(f"  → Timeline: {len(tl)} events")

    elif node_name == "run_local_rag":
        opened = update.get("opened", [])
        st.caption(f"  → Opened {len(opened)} chunks")

    elif node_name == "merge":
        st.caption("  → Results merged")


def _log_run(
    logger,
    metrics,
    graph_name: str,
    question: str,
    accumulated: dict[str, Any],
    chat_state: dict[str, Any],
    total_ms: float,
    cfg,
) -> None:
    """Log the completed run and record metrics."""
    logger.log_event(
        f"{graph_name}_completed",
        {
            "question": question[:200],
            "rewritten_query": accumulated.get("rewritten_query", "")[:200],
            "answer_length": len(accumulated.get("answer_text", "")),
            "citation_count": len(accumulated.get("citations", [])),
            "citation_map_count": len(accumulated.get("citation_map", [])),
            "retrieved_count": len(accumulated.get("retrieved", [])),
            "opened_count": len(accumulated.get("opened", [])),
            "route": accumulated.get("route"),
            "router_method": accumulated.get("router_method"),
            "subgraph_used": accumulated.get("subgraph_used"),
            "chat_turns": len(chat_state["chat_history"]),
            "tokens_in": accumulated.get("tokens_in", 0),
            "tokens_out": accumulated.get("tokens_out", 0),
            "error": accumulated.get("error"),
            "total_ms": round(total_ms, 1),
        },
    )

    metrics.record_component_latency(
        component_name=f"{graph_name}.run",
        duration_ms=total_ms,
        component_type="graph",
    )

    tokens_out = accumulated.get("tokens_out", 0)
    if tokens_out > 0:
        metrics.record_model_tokens(
            model_name=cfg.model.get("model_name", "local-model"),
            tokens_in=accumulated.get("tokens_in", 0),
            tokens_out=tokens_out,
            duration_ms=accumulated.get("model_latency_ms", 0.0),
        )

    metrics.record_retrieval(
        final_count=len(accumulated.get("opened", [])),
        query=accumulated.get("rewritten_query", question),
        keyword_candidates=len(accumulated.get("retrieved", [])),
        vector_candidates=len(accumulated.get("retrieved", [])),
        reranked=len(accumulated.get("retrieved", [])),
        duration_ms=total_ms,
    )
