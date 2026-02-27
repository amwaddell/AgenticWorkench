"""
Supervisor graph: hierarchical / multi-agent routing structure.

Routes incoming questions to the best specialist subgraph:

- **local_rag** — Wikipedia RAG (search → open → answer)
- **web_research** — web search-first path
- **timeline_research** — researcher graph with timeline + citation validation
- **math** — calculator tool for arithmetic questions

Routing logic lives in ``workbench.graphs.routing`` (heuristic rules
+ LLM fallback).  This module contains only the LangGraph node
factories, the graph builder, and the observability runner.

Topology::

    route_question ──[local_rag]──────→ run_local_rag ──→ merge → END
                   ──[web_research]───→ run_web_research ─→ merge → END
                   ──[timeline]───────→ run_timeline ────→ merge → END
                   ──[math]───────────→ run_math ────────→ merge → END

Usage::

    graph = build_supervisor_graph(
        search_tool=search_tool,
        open_chunk_tool=open_chunk_tool,
        model=llm,
        web_search_tool=web_search_tool,
        calculator_tool=calculator_tool,
    )
    result = graph.invoke({"question": "What is 17% of 340?"})

Usage (with observability)::

    result = run_supervisor_graph(graph=graph, question="…", config_snapshot=snap)
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, StateGraph

from workbench.graphs.base_state import SupervisorState
from workbench.graphs.routing import (
    DEFAULT_ROUTE,
    ROUTE_AMBIGUOUS,
    ROUTE_LOCAL_RAG,
    ROUTE_MATH,
    ROUTE_TIMELINE,
    ROUTE_WEB_RESEARCH,
    heuristic_route,
    llm_route,
)
from workbench.observability.tracing import add_span_attributes, start_span

# ------------------------------------------------------------------ #
#  Node factories                                                     #
# ------------------------------------------------------------------ #


def _make_route_node(
    model: Any | None,
    routing_mode: str = "hybrid",
    generation_settings: dict[str, Any] | None = None,
) -> Any:
    """
    Create the routing node.

    Classifies the question and writes the route + metadata into state.
    Delegates to ``heuristic_route`` and ``llm_route`` from the
    ``routing`` subpackage.
    """

    def route_node(state: SupervisorState) -> dict[str, Any]:
        question = state["question"]

        with start_span(
            "graph.supervisor.route",
            attributes={"question": question[:200], "routing_mode": routing_mode},
        ):
            t0 = time.time()

            if routing_mode == "llm_only" and model is not None:
                route, confidence = llm_route(question, model, generation_settings)
                method = "llm"
                rationale = f"LLM-only mode selected route={route}"

            elif routing_mode == "heuristic_only":
                route, confidence = heuristic_route(question)
                method = "heuristic"
                if route == ROUTE_AMBIGUOUS:
                    route = DEFAULT_ROUTE
                    confidence = 0.3
                rationale = f"Heuristic selected route={route}"

            else:
                # --- Hybrid: heuristic first, LLM fallback ---------------
                route, confidence = heuristic_route(question)
                method = "heuristic"
                rationale = f"Heuristic selected route={route} (conf={confidence:.2f})"

                if route == ROUTE_AMBIGUOUS and model is not None:
                    route, confidence = llm_route(question, model, generation_settings)
                    method = "llm"
                    rationale = f"Heuristic was ambiguous; LLM selected route={route}"
                elif route == ROUTE_AMBIGUOUS:
                    route = DEFAULT_ROUTE
                    confidence = 0.2
                    rationale = "Heuristic ambiguous, no LLM available; defaulting"

            latency_ms = (time.time() - t0) * 1000

            add_span_attributes(
                {
                    "route": route,
                    "router_method": method,
                    "router_confidence": confidence,
                    "router_latency_ms": latency_ms,
                }
            )

            return {
                "route": route,
                "router_method": method,
                "router_confidence": confidence,
                "router_rationale": rationale,
                "router_latency_ms": latency_ms,
            }

    return route_node


def _route_dispatcher(state: SupervisorState) -> str:
    """Read the route from state and return the matching node name."""
    dispatch = {
        ROUTE_LOCAL_RAG: "run_local_rag",
        ROUTE_WEB_RESEARCH: "run_web_research",
        ROUTE_TIMELINE: "run_timeline_research",
        ROUTE_MATH: "run_math",
    }
    return dispatch.get(state.get("route", DEFAULT_ROUTE), "run_local_rag")


# ------------------------------------------------------------------ #
#  Subgraph nodes                                                     #
# ------------------------------------------------------------------ #


def _make_local_rag_node(rag_subgraph: Any) -> Any:
    """Invoke the compiled rag_graph as a subgraph."""

    def run_local_rag(state: SupervisorState) -> dict[str, Any]:
        question = state["question"]

        with start_span(
            "graph.supervisor.local_rag",
            attributes={"question": question[:200]},
        ):
            try:
                sub_input: dict[str, Any] = {"question": question}
                if "search_top_k" in state:
                    sub_input["search_top_k"] = state["search_top_k"]
                if "open_top_n" in state:
                    sub_input["open_top_n"] = state["open_top_n"]

                result = rag_subgraph.invoke(sub_input)

                return {
                    "retrieved": result.get("retrieved", []),
                    "opened": result.get("opened", []),
                    "answer_text": result.get("answer_text", ""),
                    "citations": result.get("citations", []),
                    "web_citations": result.get("web_citations", []),
                    "web_search_used": result.get("web_search_used", False),
                    "web_results": result.get("web_results", []),
                    "tokens_in": result.get("tokens_in", 0),
                    "tokens_out": result.get("tokens_out", 0),
                    "model_latency_ms": result.get("model_latency_ms", 0.0),
                    "prompt_meta": result.get("prompt_meta", {}),
                    "subgraph_used": "rag_graph",
                    "error": result.get("error"),
                }

            except Exception as exc:
                return {
                    "answer_text": "",
                    "subgraph_used": "rag_graph",
                    "error": f"local_rag subgraph: {type(exc).__name__}: {exc}",
                }

    return run_local_rag


def _make_web_research_node(
    web_search_tool: Any,
    model: Any,
    prompt_builder: Any | None,
    generation_settings: dict[str, Any] | None,
) -> Any:
    """Run a web search then generate an answer from web results only."""
    from workbench.graphs.rag_graph import extract_web_citations

    gen_settings = generation_settings or {}

    fallback_system = (
        "You are a helpful research assistant.  Answer the user's question "
        "using ONLY the web search results provided below.  Cite each source "
        "by writing the URL in parentheses after the relevant claim.\n\n"
        "If the results do not contain enough information, say so honestly."
    )

    def run_web_research(state: SupervisorState) -> dict[str, Any]:
        question = state["question"]

        with start_span(
            "graph.supervisor.web_research",
            attributes={"question": question[:200]},
        ):
            try:
                search_result = web_search_tool.execute(
                    query=question,
                    max_results=5,
                )
                web_results = search_result.get("results", [])

                if not web_results:
                    return {
                        "answer_text": "Web search returned no results for this query.",
                        "web_results": [],
                        "web_search_used": True,
                        "citations": [],
                        "web_citations": [],
                        "subgraph_used": "web_research",
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "model_latency_ms": 0.0,
                        "prompt_meta": {},
                    }

                from workbench.graphs.rag_graph import _format_web_results

                web_block = _format_web_results(web_results)
                user_message = (
                    f"Web search results:\n{web_block}\n\n"
                    f"Question: {question}\n\n"
                    "Answer using the web results above.  "
                    "Cite sources by writing the URL in parentheses."
                )
                messages = [
                    {"role": "system", "content": fallback_system},
                    {"role": "user", "content": user_message},
                ]

                response = model.generate(messages, **gen_settings)

                known_urls = [r.get("url", "") for r in web_results]
                web_cites = extract_web_citations(response.text, known_urls)

                add_span_attributes(
                    {
                        "web_results_count": len(web_results),
                        "answer_length": len(response.text),
                        "web_citation_count": len(web_cites),
                    }
                )

                return {
                    "answer_text": response.text,
                    "web_results": web_results,
                    "web_search_used": True,
                    "citations": [],
                    "web_citations": web_cites,
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                    "model_latency_ms": response.latency_ms,
                    "prompt_meta": {"variant": "web_research", "template": "inline"},
                    "subgraph_used": "web_research",
                }

            except Exception as exc:
                return {
                    "answer_text": "",
                    "web_search_used": True,
                    "subgraph_used": "web_research",
                    "error": f"web_research: {type(exc).__name__}: {exc}",
                }

    return run_web_research


def _make_timeline_research_node(researcher_subgraph: Any) -> Any:
    """Invoke the compiled researcher_graph and merge its full state."""

    def run_timeline_research(state: SupervisorState) -> dict[str, Any]:
        question = state["question"]

        with start_span(
            "graph.supervisor.timeline_research",
            attributes={"question": question[:200]},
        ):
            try:
                sub_input: dict[str, Any] = {"question": question}
                if "search_top_k" in state:
                    sub_input["search_top_k"] = state["search_top_k"]
                if "open_top_n" in state:
                    sub_input["open_top_n"] = state["open_top_n"]

                result = researcher_subgraph.invoke(sub_input)

                return {
                    "retrieved": result.get("retrieved", []),
                    "opened": result.get("opened", []),
                    "answer_text": result.get("answer_text", ""),
                    "citations": result.get("citations", []),
                    "web_citations": result.get("web_citations", []),
                    "timeline": result.get("timeline", []),
                    "timeline_raw": result.get("timeline_raw", ""),
                    "citation_validation": result.get("citation_validation", {}),
                    "repair_attempts": result.get("repair_attempts", 0),
                    "tokens_in": result.get("tokens_in", 0),
                    "tokens_out": result.get("tokens_out", 0),
                    "model_latency_ms": result.get("model_latency_ms", 0.0),
                    "prompt_meta": result.get("prompt_meta", {}),
                    "subgraph_used": "researcher_graph",
                    "error": result.get("error"),
                }

            except Exception as exc:
                return {
                    "answer_text": "",
                    "subgraph_used": "researcher_graph",
                    "error": f"timeline_research subgraph: {type(exc).__name__}: {exc}",
                }

    return run_timeline_research


def _make_math_node(
    calculator_tool: Any,
    model: Any | None,
    generation_settings: dict[str, Any] | None,
) -> Any:
    """
    Calculator node.

    Tries direct evaluation first; falls back to LLM expression
    extraction for word problems.
    """
    gen_settings = generation_settings or {}

    _EXTRACT_MATH_PROMPT = (
        "Extract the mathematical expression from the following question.  "
        "Respond with ONLY the expression (no words, no equals sign).  "
        "Use Python syntax: +, -, *, /, **, sqrt(), etc.\n\n"
        "Question: {question}\n\nExpression:"
    )

    def run_math(state: SupervisorState) -> dict[str, Any]:
        question = state["question"]

        with start_span(
            "graph.supervisor.math",
            attributes={"question": question[:200]},
        ):
            try:
                # Try direct evaluation first
                q_stripped = question.strip().rstrip("?").rstrip("=").strip()
                direct_result = calculator_tool.execute(expression=q_stripped)

                if direct_result.get("result") is not None:
                    answer = (
                        f"{direct_result['expression']} = {direct_result['result']}"
                    )
                    add_span_attributes(
                        {
                            "math_method": "direct",
                            "result": str(direct_result["result"]),
                        }
                    )
                    return {
                        "answer_text": answer,
                        "calculator_result": direct_result,
                        "citations": [],
                        "web_citations": [],
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "model_latency_ms": 0.0,
                        "prompt_meta": {"variant": "math_direct"},
                        "subgraph_used": "calculator",
                    }

                # Direct eval failed — try LLM extraction
                if model is not None:
                    messages = [
                        {
                            "role": "user",
                            "content": _EXTRACT_MATH_PROMPT.format(question=question),
                        },
                    ]
                    extract_settings = {
                        **gen_settings,
                        "temperature": 0.0,
                        "max_tokens": 100,
                    }
                    response = model.generate(messages, **extract_settings)
                    extracted_expr = response.text.strip()

                    calc_result = calculator_tool.execute(expression=extracted_expr)

                    if calc_result.get("result") is not None:
                        answer = (
                            f'The answer to "{question}" is: '
                            f"{calc_result['expression']} = {calc_result['result']}"
                        )
                    else:
                        answer = (
                            f'I tried to evaluate "{extracted_expr}" but got an error: '
                            f"{calc_result.get('error', 'unknown error')}"
                        )

                    add_span_attributes(
                        {
                            "math_method": "llm_extract",
                            "extracted_expression": extracted_expr[:200],
                        }
                    )

                    return {
                        "answer_text": answer,
                        "calculator_result": calc_result,
                        "citations": [],
                        "web_citations": [],
                        "tokens_in": response.tokens_in,
                        "tokens_out": response.tokens_out,
                        "model_latency_ms": response.latency_ms,
                        "prompt_meta": {"variant": "math_llm_extract"},
                        "subgraph_used": "calculator",
                    }

                # No model, direct eval failed
                return {
                    "answer_text": (
                        f"Could not evaluate: {direct_result.get('error', 'unknown')}"
                    ),
                    "calculator_result": direct_result,
                    "subgraph_used": "calculator",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "model_latency_ms": 0.0,
                    "prompt_meta": {"variant": "math_failed"},
                }

            except Exception as exc:
                return {
                    "answer_text": f"Math evaluation failed: {exc}",
                    "subgraph_used": "calculator",
                    "error": f"math_node: {type(exc).__name__}: {exc}",
                }

    return run_math


def _make_merge_node() -> Any:
    """Pass-through node: exists for observability and as a stable merge point."""

    def merge_node(state: SupervisorState) -> dict[str, Any]:
        with start_span(
            "graph.supervisor.merge",
            attributes={
                "route": state.get("route", "unknown"),
                "subgraph_used": state.get("subgraph_used", "unknown"),
                "has_answer": bool(state.get("answer_text")),
            },
        ):
            return {}

    return merge_node


# ------------------------------------------------------------------ #
#  Graph builder                                                      #
# ------------------------------------------------------------------ #


def build_supervisor_graph(
    search_tool: Any,
    open_chunk_tool: Any,
    model: Any,
    prompt_builder: Any | None = None,
    generation_settings: dict[str, Any] | None = None,
    open_top_n: int = 3,
    search_top_k: int = 5,
    web_search_tool: Any | None = None,
    confidence_threshold: int = 2,
    timeline_tool: Any | None = None,
    calculator_tool: Any | None = None,
    routing_mode: str = "hybrid",
) -> Any:
    """
    Build and compile the supervisor (hierarchical) graph.

    Routes that lack required tools gracefully fall back to ``local_rag``.

    Args:
        search_tool:  SearchWikipediaTool or equivalent.
        open_chunk_tool:  OpenChunkTool or equivalent.
        model:  LanguageModel.
        prompt_builder:  Optional ``PromptBuilder``.
        generation_settings:  Extra kwargs for ``model.generate()``.
        open_top_n:  Default chunks to open.
        search_top_k:  Default chunks to retrieve.
        web_search_tool:  Optional ``WebSearchTool``.
        confidence_threshold:  For the RAG subgraph's web fallback.
        timeline_tool:  Optional ``TimelineTool``.
        calculator_tool:  Optional ``CalculatorTool``.
        routing_mode:  ``"hybrid"`` | ``"heuristic_only"`` | ``"llm_only"``.

    Returns:
        Compiled LangGraph ready for ``.invoke()`` and ``.stream()``.
    """
    # --- Build subgraphs ----------------------------------------------
    from workbench.graphs.rag_graph import build_rag_graph

    rag_subgraph = build_rag_graph(
        search_tool=search_tool,
        open_chunk_tool=open_chunk_tool,
        model=model,
        prompt_builder=prompt_builder,
        generation_settings=generation_settings,
        open_top_n=open_top_n,
        search_top_k=search_top_k,
        web_search_tool=web_search_tool,
        confidence_threshold=confidence_threshold,
    )

    researcher_subgraph = None
    if timeline_tool is not None:
        from workbench.graphs.researcher_graph import build_researcher_graph

        researcher_subgraph = build_researcher_graph(
            search_tool=search_tool,
            open_chunk_tool=open_chunk_tool,
            timeline_tool=timeline_tool,
            model=model,
            prompt_builder=prompt_builder,
            generation_settings=generation_settings,
            open_top_n=open_top_n,
            search_top_k=search_top_k,
        )

    # --- Build supervisor StateGraph ----------------------------------
    builder: StateGraph = StateGraph(SupervisorState)

    builder.add_node(
        "route_question",
        _make_route_node(model, routing_mode, generation_settings),
    )
    builder.add_node("run_local_rag", _make_local_rag_node(rag_subgraph))

    if web_search_tool is not None:
        builder.add_node(
            "run_web_research",
            _make_web_research_node(
                web_search_tool,
                model,
                prompt_builder,
                generation_settings,
            ),
        )

    if researcher_subgraph is not None:
        builder.add_node(
            "run_timeline_research",
            _make_timeline_research_node(researcher_subgraph),
        )

    if calculator_tool is not None:
        builder.add_node(
            "run_math",
            _make_math_node(calculator_tool, model, generation_settings),
        )

    builder.add_node("merge", _make_merge_node())

    # --- Wire edges ---------------------------------------------------
    builder.set_entry_point("route_question")

    dispatch_map: dict[str, str] = {"run_local_rag": "run_local_rag"}
    if web_search_tool is not None:
        dispatch_map["run_web_research"] = "run_web_research"
    if researcher_subgraph is not None:
        dispatch_map["run_timeline_research"] = "run_timeline_research"
    if calculator_tool is not None:
        dispatch_map["run_math"] = "run_math"

    def safe_dispatcher(state: SupervisorState) -> str:
        target = _route_dispatcher(state)
        return target if target in dispatch_map else "run_local_rag"

    builder.add_conditional_edges("route_question", safe_dispatcher, dispatch_map)

    builder.add_edge("run_local_rag", "merge")
    if web_search_tool is not None:
        builder.add_edge("run_web_research", "merge")
    if researcher_subgraph is not None:
        builder.add_edge("run_timeline_research", "merge")
    if calculator_tool is not None:
        builder.add_edge("run_math", "merge")

    builder.add_edge("merge", END)

    return builder.compile()


# ------------------------------------------------------------------ #
#  Convenience runner (observability wrapper)                          #
# ------------------------------------------------------------------ #


def run_supervisor_graph(
    graph: Any,
    question: str,
    config_snapshot: dict[str, Any] | None = None,
    *,
    search_top_k: int | None = None,
    open_top_n: int | None = None,
    run_type: str = "supervisor_graph",
    service_name: str = "agentic-workbench-supervisor-graph",
) -> SupervisorState:
    """
    Invoke the supervisor graph with full observability setup.

    Args:
        graph:  Compiled supervisor graph.
        question:  The user question.
        config_snapshot:  Frozen config dict for the run record.
        search_top_k:  Override the default search_top_k.
        open_top_n:  Override the default open_top_n.
        run_type:  Run type label for the context.
        service_name:  Service name for tracing.

    Returns:
        Final ``SupervisorState`` dict.
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
            "supervisor_graph.run",
            attributes={"question": question[:200]},
        ):
            t0 = time.time()

            initial_state: dict[str, Any] = {"question": question}
            if search_top_k is not None:
                initial_state["search_top_k"] = search_top_k
            if open_top_n is not None:
                initial_state["open_top_n"] = open_top_n

            result: SupervisorState = graph.invoke(initial_state)

            total_ms = (time.time() - t0) * 1000

        # --- Log results ----------------------------------------------
        logger.log_event(
            "supervisor_graph_completed",
            {
                "question": question[:200],
                "route": result.get("route", "unknown"),
                "router_method": result.get("router_method", "unknown"),
                "router_confidence": result.get("router_confidence", 0.0),
                "router_rationale": result.get("router_rationale", ""),
                "router_latency_ms": result.get("router_latency_ms", 0.0),
                "subgraph_used": result.get("subgraph_used", "unknown"),
                "answer_length": len(result.get("answer_text", "")),
                "citation_count": len(result.get("citations", [])),
                "web_citation_count": len(result.get("web_citations", [])),
                "retrieved_count": len(result.get("retrieved", [])),
                "opened_count": len(result.get("opened", [])),
                "tokens_in": result.get("tokens_in", 0),
                "tokens_out": result.get("tokens_out", 0),
                "error": result.get("error"),
                "total_ms": round(total_ms, 1),
            },
        )

        tokens_out = result.get("tokens_out", 0)
        if tokens_out > 0:
            logger.log_model_call(
                model_name="local-model",
                tokens_in=result.get("tokens_in", 0),
                tokens_out=tokens_out,
                duration_ms=result.get("model_latency_ms", 0.0),
            )

        # --- Record metrics -------------------------------------------
        metrics.record_component_latency(
            component_name="supervisor_graph.run",
            duration_ms=total_ms,
            component_type="graph",
        )
        metrics.record_component_latency(
            component_name="supervisor_graph.router",
            duration_ms=result.get("router_latency_ms", 0.0),
            component_type="router",
        )

        if tokens_out > 0:
            metrics.record_model_tokens(
                model_name="local-model",
                tokens_in=result.get("tokens_in", 0),
                tokens_out=tokens_out,
                duration_ms=result.get("model_latency_ms", 0.0),
            )

        if result.get("retrieved"):
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
