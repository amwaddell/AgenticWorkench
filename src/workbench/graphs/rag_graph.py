"""
RAG graph: search → open → answer as a compiled LangGraph.

This graph replaces the sequential ``AgentLoop`` + ``ScriptedPolicy``
combination with an explicit DAG.

**Day 5 addition**: an optional web search fallback.  When
``web_search_tool`` is provided to ``build_rag_graph()``, the graph
adds a confidence-check node after retrieval.  If retrieval returns
fewer results than ``confidence_threshold``, the graph branches to
a web search node before proceeding to the answer.

**Day 9 addition**: multi-turn chat support.  When ``chat_history``
is present in the initial state, the graph inserts a ``rewrite_query``
node before retrieval to resolve coreferences (e.g. "that battle" →
"the Battle of Stalingrad").  The answer node also receives
conversation context so the LLM can produce coherent follow-up replies.

Topology (single-turn, no web search)::

    retrieve  →  open  →  answer  →  END

Topology (multi-turn, no web search)::

    rewrite_query  →  retrieve  →  open  →  answer  →  END

Topology (multi-turn, with web search)::

    rewrite_query → retrieve → check_confidence ─[sufficient]─→ open → answer → END
                                                  │
                                              [low_confidence]
                                                  │
                                             web_search → open → answer → END

Usage (single-turn — backward compatible)::

    graph = build_rag_graph(search_tool, open_chunk_tool, model)
    result = graph.invoke({"question": "When did the Roman Republic end?"})

Usage (multi-turn chat)::

    result = graph.invoke({
        "question": "What was that battle about?",
        "chat_history": [
            {"role": "user", "content": "Tell me about WW2"},
            {"role": "assistant", "content": "World War II was a global conflict..."},
        ],
        "conversation_summary": "Discussing WW2 and the Eastern Front.",
    })
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
    format_conversation_context,
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


def extract_web_citations(answer_text: str, urls: list[str]) -> list[str]:
    """
    Find URLs mentioned in the answer text.

    Args:
        answer_text: The generated answer string.
        urls: List of URLs from web search results.

    Returns:
        List of cited URLs found in the text.
    """
    cited: list[str] = []
    for url in urls:
        if url in answer_text:
            cited.append(url)
    return cited


# ------------------------------------------------------------------ #
#  Node factories (closures over injected dependencies)               #
# ------------------------------------------------------------------ #


def _make_rewrite_query_node(
    model: Any,
    prompt_builder: PromptBuilder | None,
    generation_settings: dict[str, Any] | None,
) -> Any:
    """
    Create the query rewrite node for multi-turn conversations.

    When ``chat_history`` is present in state, this node calls the LLM
    to resolve coreferences and produce a standalone search query.

    When there is no chat history (first turn), the node passes the
    original question through unchanged.
    """
    gen_settings = generation_settings or {"temperature": 0.1, "max_tokens": 100}

    # Fallback prompt for when no PromptBuilder is available
    _FALLBACK_REWRITE_PROMPT = (
        "Rewrite the following question as a standalone search query, "
        "resolving any pronouns or references using the conversation context. "
        "Output ONLY the rewritten query.\n\n"
        "{conversation_context}\n"
        "Latest question: {question}\n\n"
        "Standalone query:"
    )

    def rewrite_query_node(state: RAGState) -> dict[str, Any]:
        chat_history = state.get("chat_history", [])
        conversation_summary = state.get("conversation_summary", "")
        question = state["question"]

        # No history → skip rewrite, use original question
        if not chat_history:
            return {"rewritten_query": question}

        with start_span(
            "graph.rewrite_query_node",
            attributes={
                "question": question[:200],
                "chat_turns": len(chat_history),
                "has_summary": bool(conversation_summary),
            },
        ):
            try:
                conversation_context = format_conversation_context(
                    chat_history=chat_history,
                    conversation_summary=conversation_summary,
                )

                if prompt_builder is not None:
                    try:
                        user_prompt = prompt_builder.render(
                            "chat_query_rewrite",
                            question=question,
                            conversation_context=conversation_context,
                        )
                    except FileNotFoundError:
                        # Template not found — use fallback
                        user_prompt = _FALLBACK_REWRITE_PROMPT.format(
                            question=question,
                            conversation_context=conversation_context,
                        )
                else:
                    user_prompt = _FALLBACK_REWRITE_PROMPT.format(
                        question=question,
                        conversation_context=conversation_context,
                    )

                messages = [{"role": "user", "content": user_prompt}]
                response = model.generate(messages, **gen_settings)

                rewritten = response.text.strip()
                # Safety: if the model returns something empty or too long,
                # fall back to original question
                if not rewritten or len(rewritten) > 300:
                    rewritten = question

                add_span_attributes(
                    {
                        "rewritten_query": rewritten[:200],
                        "original_question": question[:200],
                    }
                )

                return {"rewritten_query": rewritten}

            except Exception as exc:
                # Rewrite failure is non-fatal — use original question
                add_span_attributes({"rewrite_error": str(exc)[:200]})
                return {"rewritten_query": question}

    return rewrite_query_node


def _make_retrieve_node(
    search_tool: Any,
    default_top_k: int,
) -> Any:
    """
    Create the retrieve node function.

    The node calls ``search_tool.execute()`` with the question and
    writes the result into ``state["retrieved"]``.

    In multi-turn mode, uses ``rewritten_query`` instead of the raw
    ``question`` for retrieval.
    """

    def retrieve_node(state: RAGState) -> dict[str, Any]:
        # Prefer the rewritten query (from rewrite node) over raw question
        query = state.get("rewritten_query") or state["question"]
        top_k = state.get("search_top_k", default_top_k)

        with start_span(
            "graph.retrieve_node",
            attributes={"question": query[:200], "top_k": top_k},
        ):
            try:
                result = search_tool.execute(query=query, top_k=top_k)
                chunks = result.get("chunks", [])

                add_span_attributes({"results_count": len(chunks)})

                return {"retrieved": chunks}

            except Exception as exc:
                return {
                    "retrieved": [],
                    "error": f"retrieve_node: {type(exc).__name__}: {exc}",
                }

    return retrieve_node


def _make_check_confidence_node(threshold: int) -> Any:
    """
    Create a confidence-check node.

    If retrieval returned fewer than ``threshold`` results, this
    signals that a web search fallback should be used.  The node
    itself doesn't branch — it just writes ``web_search_used``
    into state.  The conditional edge reads that flag.
    """

    def check_confidence_node(state: RAGState) -> dict[str, Any]:
        retrieved = state.get("retrieved", [])
        needs_web = len(retrieved) < threshold

        with start_span(
            "graph.check_confidence",
            attributes={
                "retrieved_count": len(retrieved),
                "threshold": threshold,
                "needs_web_search": needs_web,
            },
        ):
            return {"web_search_used": needs_web}

    return check_confidence_node


def _make_web_search_node(web_search_tool: Any) -> Any:
    """
    Create the web search fallback node.

    Calls the web search tool with the question and writes web
    results into state.  These are combined with any Wikipedia
    results in the answer node.

    In multi-turn mode, uses ``rewritten_query`` for better web search.
    """

    def web_search_node(state: RAGState) -> dict[str, Any]:
        # Use rewritten query for web search too
        query = state.get("rewritten_query") or state["question"]

        with start_span(
            "graph.web_search_node",
            attributes={"question": query[:200]},
        ):
            try:
                result = web_search_tool.execute(query=query, max_results=5)
                web_results = result.get("results", [])

                add_span_attributes(
                    {
                        "web_results_count": len(web_results),
                        "provider": result.get("provider", "unknown"),
                    }
                )

                return {"web_results": web_results}

            except Exception as exc:
                # Web search failure is non-fatal
                return {
                    "web_results": [],
                    "error": f"web_search_node: {type(exc).__name__}: {exc}",
                }

    return web_search_node


def _confidence_router(state: RAGState) -> str:
    """Route based on whether web search is needed."""
    if state.get("web_search_used", False):
        return "low_confidence"
    return "sufficient"


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

    The node builds a prompt from the opened evidence (and optional
    web search results), calls the language model, extracts citations,
    and writes the answer + metadata into state.

    In multi-turn mode, includes conversation context in the prompt
    so the LLM can produce coherent follow-up replies.
    """
    gen_settings = generation_settings or {}

    # Fallback system prompt — used when no PromptBuilder is provided.
    fallback_system = (
        "You are a helpful research assistant. Answer the user's question "
        "using ONLY the evidence provided below. For each claim, cite the "
        "source by writing [chunk_id] immediately after the claim.\n\n"
        "If web search results are included, cite them by writing the URL "
        "in parentheses after the claim.\n\n"
        "If the evidence does not contain enough information, say so honestly."
    )

    def answer_node(state: RAGState) -> dict[str, Any]:
        opened_dicts = state.get("opened", [])
        web_results = state.get("web_results", [])
        question = state["question"]
        chat_history = state.get("chat_history", [])
        conversation_summary = state.get("conversation_summary", "")

        # If upstream failed, propagate
        if state.get("error"):
            return {}

        # No evidence at all
        if not opened_dicts and not web_results:
            return {
                "answer_text": (
                    "I don't have enough evidence in the provided "
                    "articles to fully answer this question."
                ),
                "citations": [],
                "web_citations": [],
                "tokens_in": 0,
                "tokens_out": 0,
                "model_latency_ms": 0.0,
                "prompt_meta": {},
            }

        with start_span(
            "graph.answer_node",
            attributes={
                "evidence_count": len(opened_dicts),
                "web_results_count": len(web_results),
                "chat_turns": len(chat_history),
            },
        ):
            try:
                # --- Choose template variant ----------------------
                has_history = bool(chat_history)

                if prompt_builder is not None and opened_dicts:
                    # Pick chat-aware template if we have history
                    if has_history:
                        try:
                            variant = "chat_answer_with_citations"
                            messages, meta = prompt_builder.build(
                                variant,
                                question=question,
                                opened_chunks=opened_dicts,
                                chat_history=chat_history,
                                conversation_summary=conversation_summary,
                            )
                        except FileNotFoundError:
                            # Chat template not installed — fall back
                            variant = "answer_with_citations"
                            messages, meta = prompt_builder.build(
                                variant,
                                question=question,
                                opened_chunks=opened_dicts,
                            )
                    else:
                        variant = "answer_with_citations"
                        messages, meta = prompt_builder.build(
                            variant,
                            question=question,
                            opened_chunks=opened_dicts,
                        )

                    # Append web results to user message if present
                    if web_results:
                        web_block = _format_web_results(web_results)
                        messages[-1]["content"] += (
                            f"\n\nAdditional web search results:\n{web_block}"
                        )
                        meta["web_results_included"] = len(web_results)
                else:
                    # Fallback: inline prompt
                    evidence_parts: list[str] = []

                    if opened_dicts:
                        opened_objs = [OpenedChunk(**d) for d in opened_dicts]
                        evidence_parts.append(
                            f"Wikipedia evidence:\n{format_evidence_block(opened_objs)}"
                        )

                    if web_results:
                        evidence_parts.append(
                            f"Web search results:\n{_format_web_results(web_results)}"
                        )

                    # Include conversation context in fallback too
                    if has_history:
                        conv_ctx = format_conversation_context(
                            chat_history=chat_history,
                            conversation_summary=conversation_summary,
                        )
                        evidence_parts.insert(0, conv_ctx)

                    user_message = (
                        "\n\n".join(evidence_parts) + f"\n\nQuestion: {question}\n\n"
                        "Answer the question using the evidence above. "
                        "Cite sources using [chunk_id] for Wikipedia chunks "
                        "and (URL) for web results."
                    )
                    messages = [
                        {"role": "system", "content": fallback_system},
                        {"role": "user", "content": user_message},
                    ]
                    meta = {"variant": "fallback", "template": "inline"}
                    if web_results:
                        meta["web_results_included"] = len(web_results)

                # --- Call model -----------------------------------
                response = model.generate(messages, **gen_settings)

                # --- Extract citations ----------------------------
                known_ids = [d.get("chunk_id", "") for d in opened_dicts]
                citations = extract_citations(response.text, known_ids)

                known_urls = [r.get("url", "") for r in web_results]
                web_cites = extract_web_citations(response.text, known_urls)

                add_span_attributes(
                    {
                        "answer_length": len(response.text),
                        "citation_count": len(citations),
                        "web_citation_count": len(web_cites),
                        "tokens_in": response.tokens_in,
                        "tokens_out": response.tokens_out,
                    }
                )

                return {
                    "answer_text": response.text,
                    "citations": citations,
                    "web_citations": web_cites,
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                    "model_latency_ms": response.latency_ms,
                    "prompt_meta": meta,
                }

            except Exception as exc:
                return {
                    "answer_text": "",
                    "citations": [],
                    "web_citations": [],
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "model_latency_ms": 0.0,
                    "prompt_meta": {},
                    "error": f"answer_node: {type(exc).__name__}: {exc}",
                }

    return answer_node


def _format_web_results(results: list[dict[str, Any]]) -> str:
    """Format web search results into a text block for the prompt."""
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        lines.append(f"[Web {i}] {title}\nURL: {url}\n{snippet}")
    return "\n\n".join(lines)


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
    web_search_tool: Any | None = None,
    confidence_threshold: int = 2,
) -> Any:
    """
    Build and compile the RAG graph.

    The graph always includes a ``rewrite_query`` node as the entry
    point.  On the first turn (no ``chat_history``), the rewrite node
    is a no-op that copies ``question`` → ``rewritten_query``.  On
    follow-up turns, it calls the LLM to resolve coreferences.

    Without ``web_search_tool``::

        rewrite_query → retrieve → open → answer → END

    With ``web_search_tool``::

        rewrite_query → retrieve → check_confidence ─[sufficient]─→ open → answer → END
                                                      │
                                                  [low_confidence]
                                                      │
                                                 web_search → open → answer → END

    Args:
        search_tool:  SearchWikipediaTool or equivalent.
        open_chunk_tool:  OpenChunkTool or equivalent.
        model:  LanguageModel with ``.generate(messages, **settings)``.
        prompt_builder:  Optional ``PromptBuilder`` instance.
        generation_settings:  Extra kwargs for ``model.generate()``.
        open_top_n:  Default chunks to open.
        search_top_k:  Default chunks to retrieve.
        web_search_tool:  Optional ``WebSearchTool`` instance.  When
            provided, enables the confidence-check + web search fallback.
        confidence_threshold:  Minimum number of retrieval results
            needed to skip web search (default 2).

    Returns:
        A compiled LangGraph (``CompiledGraph``) ready for ``.invoke()``
        or ``.stream()``.
    """
    builder: StateGraph = StateGraph(RAGState)

    # --- Register nodes -------------------------------------------
    builder.add_node(
        "rewrite_query",
        _make_rewrite_query_node(model, prompt_builder, generation_settings),
    )
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

    # --- Entry point: always start with rewrite -------------------
    builder.set_entry_point("rewrite_query")
    builder.add_edge("rewrite_query", "retrieve")

    if web_search_tool is not None:
        # --- With web search fallback ---
        builder.add_node(
            "check_confidence",
            _make_check_confidence_node(confidence_threshold),
        )
        builder.add_node(
            "web_search",
            _make_web_search_node(web_search_tool),
        )

        builder.add_edge("retrieve", "check_confidence")
        builder.add_conditional_edges(
            "check_confidence",
            _confidence_router,
            {
                "sufficient": "open",
                "low_confidence": "web_search",
            },
        )
        builder.add_edge("web_search", "open")
    else:
        # --- Without web search ---
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
    chat_history: list[dict[str, str]] | None = None,
    conversation_summary: str | None = None,
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
        chat_history:  Recent conversation turns for multi-turn chat.
        conversation_summary:  Rolling summary of older conversation.
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
            if chat_history is not None:
                initial_state["chat_history"] = chat_history
            if conversation_summary is not None:
                initial_state["conversation_summary"] = conversation_summary

            # --- Invoke -------------------------------------------
            result: RAGState = graph.invoke(initial_state)

            total_ms = (time.time() - t0) * 1000

        # --- Log results ------------------------------------------
        logger.log_event(
            "rag_graph_completed",
            {
                "question": question[:200],
                "rewritten_query": result.get("rewritten_query", "")[:200],
                "answer_length": len(result.get("answer_text", "")),
                "citation_count": len(result.get("citations", [])),
                "web_citation_count": len(result.get("web_citations", [])),
                "retrieved_count": len(result.get("retrieved", [])),
                "opened_count": len(result.get("opened", [])),
                "web_search_used": result.get("web_search_used", False),
                "web_results_count": len(result.get("web_results", [])),
                "chat_turns": len(chat_history) if chat_history else 0,
                "tokens_in": result.get("tokens_in", 0),
                "tokens_out": result.get("tokens_out", 0),
                "error": result.get("error"),
                "total_ms": round(total_ms, 1),
            },
        )

        if result.get("retrieved"):
            logger.log_retrieval(
                query=result.get("rewritten_query", question),
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
            query=result.get("rewritten_query", question),
            keyword_candidates=len(result.get("retrieved", [])),
            vector_candidates=len(result.get("retrieved", [])),
            reranked=len(result.get("retrieved", [])),
            duration_ms=total_ms,
        )

        print(f"\nRun ID: {ctx.run_id}")
        return result
