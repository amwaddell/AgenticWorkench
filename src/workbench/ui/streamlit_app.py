"""
Streamlit local chat UI for the agentic workbench.

Features
--------
* ChatGPT-style conversation interface with multi-turn context
* Query rewriting for follow-up questions (resolves "that", "it", etc.)
* System selector: ``rag_graph``, ``researcher_graph``, or ``supervisor_graph``
* Node-level streaming — shows live progress as the graph executes
* Citations panel with chunk IDs and web URLs
* Timeline display (researcher and supervisor graphs)
* Routing info display (supervisor graph)
* Run ID + Phoenix trace link for every turn
* Conversation summary for long sessions

Launch::

    streamlit run src/workbench/ui/streamlit_app.py

Or::

    bash scripts/run_ui.sh
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st

# ------------------------------------------------------------------ #
#  Page config (must be the first Streamlit call)                      #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Agentic Workbench",
    page_icon="🔬",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

# Maximum recent turns to keep in full before summarising older ones.
# Each "turn" is one user + one assistant message.
_MAX_RECENT_TURNS = 4  # 4 pairs = 8 messages


# ------------------------------------------------------------------ #
#  Imports — deferred so Streamlit can show a spinner while loading    #
# ------------------------------------------------------------------ #


@st.cache_resource(show_spinner="Loading configuration…")
def _load_config():
    """Load workbench config once."""
    from workbench.core.config import load_config

    return load_config()


@st.cache_resource(show_spinner="Loading language model client…")
def _build_model(_cfg):
    """Build the LLM client (cheap — just an HTTP wrapper)."""
    from workbench.models.llamacpp_server import LlamaCppServerModel

    return LlamaCppServerModel(
        base_url=_cfg.model["base_url"],
        model_name=_cfg.model.get("model_name", "local-model"),
        timeout_seconds=_cfg.model.get("timeout_seconds", 60),
        default_temperature=_cfg.model.get("temperature", 0.7),
        default_max_tokens=_cfg.model.get("max_tokens", 2048),
    )


@st.cache_resource(show_spinner="Connecting to embedding service…")
def _build_embedder(_cfg):
    """
    Build embedder from config.

    With ``provider: http`` this is instant (just creates an HTTP client).
    With ``provider: local`` this loads weights in-process (slow).
    """
    from workbench.systems.runner import build_embedder

    return build_embedder(_cfg)


@st.cache_resource(show_spinner="Connecting to reranker service…")
def _build_reranker(_cfg):
    """
    Build reranker from config (or None if disabled).

    With ``provider: http`` this is instant (just creates an HTTP client).
    With ``provider: cross_encoder`` this loads weights in-process (slow).
    """
    from workbench.systems.runner import build_reranker

    return build_reranker(_cfg)


@st.cache_resource(show_spinner="Opening chunk store…")
def _build_chunk_store(_cfg):
    """Build the ChunkStore (lazy LanceDB table load)."""
    from workbench.stores.chunk_store import ChunkStore

    db_path = Path(_cfg.paths["active_index"]) / "lancedb"
    return ChunkStore(db_path=db_path)


@st.cache_resource(show_spinner="Building retrieval pipeline…")
def _build_retriever(_cfg, _embedder, _reranker, _chunk_store):
    """Assemble the hybrid retriever."""
    from workbench.retrieval.hybrid import HybridRetriever
    from workbench.retrieval.keyword_search import LanceDBKeywordSearcher
    from workbench.retrieval.vector_search import LanceDBVectorSearcher

    db_path = Path(_cfg.paths["active_index"]) / "lancedb"
    keyword_searcher = LanceDBKeywordSearcher(db_path=db_path)
    vector_searcher = LanceDBVectorSearcher(db_path=db_path, embedder=_embedder)

    rcfg = _cfg.retrieval
    return HybridRetriever(
        keyword_searcher=keyword_searcher,
        vector_searcher=vector_searcher,
        reranker=_reranker,
        chunk_store=_chunk_store,
        keyword_top_k=rcfg.get("keyword_top_k", 20),
        vector_top_k=rcfg.get("vector_top_k", 20),
        merge_top_n=rcfg.get("merge_top_n", 50),
        rrf_k=rcfg.get("rrf_k", 60),
    )


@st.cache_resource(show_spinner="Compiling tools…")
def _build_tools(_cfg, _retriever, _chunk_store, _model):
    """Build all tools and return them in a dict."""
    from workbench.prompting.prompt_builder import PromptBuilder
    from workbench.tools.open_chunk import OpenChunkTool
    from workbench.tools.search_wikipedia import SearchWikipediaTool
    from workbench.tools.timeline import TimelineTool

    tools: dict[str, Any] = {
        "search": SearchWikipediaTool(retriever=_retriever),
        "open_chunk": OpenChunkTool(chunk_store=_chunk_store),
        "timeline": TimelineTool(
            model=_model,
            prompt_builder=PromptBuilder(),
        ),
    }

    # Optional web search tool
    if _cfg.web_search.get("enabled", False):
        try:
            from workbench.tools.web_search import WebSearchTool

            tools["web_search"] = WebSearchTool(
                provider=_cfg.web_search.get("provider", "duckduckgo"),
            )
        except Exception:
            pass  # web search not available — skip silently

    # Calculator tool (Day 5)
    try:
        from workbench.tools.utility.calculator import CalculatorTool

        tools["calculator"] = CalculatorTool()
    except Exception:
        pass

    return tools


def _build_graph(graph_name: str, cfg, model, tools):
    """Build a specific compiled graph (not cached — lightweight)."""
    from workbench.prompting.prompt_builder import PromptBuilder

    prompt_builder = PromptBuilder()

    if graph_name == "rag_graph":
        from workbench.graphs.rag_graph import build_rag_graph

        return build_rag_graph(
            search_tool=tools["search"],
            open_chunk_tool=tools["open_chunk"],
            model=model,
            prompt_builder=prompt_builder,
            open_top_n=cfg.retrieval.get("final_top_k", 3),
            search_top_k=cfg.retrieval.get("keyword_top_k", 5),
            web_search_tool=tools.get("web_search"),
            confidence_threshold=cfg.web_search.get("confidence_threshold", 2),
        )

    elif graph_name == "researcher_graph":
        from workbench.graphs.researcher_graph import build_researcher_graph

        return build_researcher_graph(
            search_tool=tools["search"],
            open_chunk_tool=tools["open_chunk"],
            timeline_tool=tools["timeline"],
            model=model,
            prompt_builder=prompt_builder,
            open_top_n=cfg.retrieval.get("final_top_k", 5),
            search_top_k=cfg.retrieval.get("keyword_top_k", 10),
        )

    elif graph_name == "supervisor_graph":
        from workbench.graphs.supervisor_graph import build_supervisor_graph

        # Read routing mode from config (default: hybrid)
        supervisor_cfg = getattr(cfg, "supervisor", {}) or {}
        routing_mode = supervisor_cfg.get("routing_mode", "hybrid")

        return build_supervisor_graph(
            search_tool=tools["search"],
            open_chunk_tool=tools["open_chunk"],
            model=model,
            prompt_builder=prompt_builder,
            open_top_n=cfg.retrieval.get("final_top_k", 3),
            search_top_k=cfg.retrieval.get("keyword_top_k", 5),
            web_search_tool=tools.get("web_search"),
            confidence_threshold=cfg.web_search.get("confidence_threshold", 2),
            timeline_tool=tools.get("timeline"),
            calculator_tool=tools.get("calculator"),
            routing_mode=routing_mode,
        )

    else:
        raise ValueError(f"Unknown graph: {graph_name}")


# ------------------------------------------------------------------ #
#  Chat history management                                             #
# ------------------------------------------------------------------ #


def _build_chat_state() -> dict[str, Any]:
    """
    Build the chat_history and conversation_summary fields from
    Streamlit session state for injection into the graph.

    Policy:
    - Keep the last ``_MAX_RECENT_TURNS`` user+assistant pairs verbatim.
    - Older turns are represented only by ``conversation_summary``
      (which is updated each time we evict turns).

    Returns:
        Dict with ``chat_history`` and ``conversation_summary`` keys.
    """
    messages = st.session_state.get("messages", [])

    # Extract user/assistant pairs
    chat_turns: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            chat_turns.append({"role": role, "content": content})

    # No history → first turn
    if not chat_turns:
        return {"chat_history": [], "conversation_summary": ""}

    # Split into "old" (to summarise) and "recent" (to keep verbatim)
    max_messages = _MAX_RECENT_TURNS * 2  # pairs → individual messages
    recent = chat_turns[-max_messages:]

    # Keep the existing rolling summary
    conversation_summary = st.session_state.get("conversation_summary", "")

    return {
        "chat_history": recent,
        "conversation_summary": conversation_summary,
    }


def _update_conversation_summary(model, cfg) -> None:
    """
    If the conversation has grown beyond the recent-turn window,
    generate a rolling summary of the older turns.

    This is called after each assistant response is added.
    """
    messages = st.session_state.get("messages", [])
    max_messages = _MAX_RECENT_TURNS * 2

    # Only summarise when we have more turns than the window
    if len(messages) <= max_messages + 2:
        return

    # Gather the turns that are about to be evicted from the window
    # (the ones just before the recent window)
    evicted = messages[-(max_messages + 2) : -max_messages]
    if not evicted:
        return

    existing_summary = st.session_state.get("conversation_summary", "")

    # Build a simple summary prompt
    evicted_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content'][:300]}" for m in evicted
    )

    summary_prompt = (
        "Summarise the following conversation turns into 2-3 sentences. "
        "Preserve key topics, entities, and conclusions.\n\n"
    )
    if existing_summary:
        summary_prompt += f"Previous summary: {existing_summary}\n\n"
    summary_prompt += f"New turns:\n{evicted_text}\n\nUpdated summary:"

    try:
        response = model.generate(
            [{"role": "user", "content": summary_prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        st.session_state["conversation_summary"] = response.text.strip()
    except Exception:
        # Summary failure is non-critical — keep existing summary
        pass


# ------------------------------------------------------------------ #
#  Human-readable labels for graph node events                         #
# ------------------------------------------------------------------ #

_NODE_LABELS: dict[str, str] = {
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
    # Supervisor nodes (Day 7)
    "route_question": "🧭 Routing question to specialist…",
    "run_local_rag": "📚 Running local RAG pipeline…",
    "run_web_research": "🌐 Running web research…",
    "run_timeline_research": "📅 Running timeline researcher…",
    "run_math": "🔢 Evaluating math expression…",
    "merge": "🔀 Merging results…",
}


# ------------------------------------------------------------------ #
#  Rendering helpers                                                   #
# ------------------------------------------------------------------ #


def _render_citations(result: dict[str, Any]) -> None:
    """Render a citations panel below the answer."""
    citations = result.get("citations", [])
    web_citations = result.get("web_citations", [])

    if not citations and not web_citations:
        return

    with st.expander(
        f"📚 Citations ({len(citations)} chunks, {len(web_citations)} web)",
        expanded=False,
    ):
        if citations:
            st.markdown("**Chunk citations:**")
            for cid in citations:
                st.code(cid, language=None)
        if web_citations:
            st.markdown("**Web citations:**")
            for url in web_citations:
                st.markdown(f"- [{url}]({url})")


def _render_timeline(result: dict[str, Any]) -> None:
    """Render timeline items from the researcher graph."""
    timeline = result.get("timeline", [])
    if not timeline:
        return

    with st.expander(f"📅 Timeline ({len(timeline)} events)", expanded=False):
        for item in timeline:
            date = item.get("date", "?")
            event = item.get("event", "")
            chunks = item.get("supporting_chunk_ids", [])
            cite_str = ", ".join(f"`{c}`" for c in chunks) if chunks else ""
            st.markdown(f"**{date}** — {event}")
            if cite_str:
                st.caption(f"Sources: {cite_str}")


def _render_routing_info(result: dict[str, Any]) -> None:
    """Render routing decision info from the supervisor graph."""
    route = result.get("route")
    if not route:
        return

    with st.expander("🧭 Routing decision", expanded=False):
        cols = st.columns(3)
        cols[0].metric("Route", route)
        cols[1].metric("Method", result.get("router_method", "?"))
        cols[2].metric(
            "Confidence",
            f"{result.get('router_confidence', 0):.0%}",
        )

        rationale = result.get("router_rationale", "")
        if rationale:
            st.caption(rationale)

        subgraph = result.get("subgraph_used", "")
        if subgraph:
            st.caption(f"Subgraph used: `{subgraph}`")

        router_ms = result.get("router_latency_ms", 0)
        if router_ms > 0:
            st.caption(f"Router latency: {router_ms:.1f} ms")


def _render_run_metadata(result: dict[str, Any], run_id: str, cfg) -> None:
    """Render run metadata (tokens, latency, run ID, Phoenix link)."""
    tokens_in = result.get("tokens_in", 0)
    tokens_out = result.get("tokens_out", 0)
    latency = result.get("model_latency_ms", 0.0)
    phoenix_endpoint = cfg.observability.get(
        "phoenix_endpoint", "http://localhost:6006"
    )

    # Show rewritten query if different from original
    rewritten = result.get("rewritten_query", "")
    question = result.get("question", "")
    if rewritten and rewritten != question:
        st.caption(f"🔄 Rewritten query: *{rewritten}*")

    cols = st.columns(4)
    cols[0].metric("Tokens in", tokens_in)
    cols[1].metric("Tokens out", tokens_out)
    cols[2].metric("Model latency", f"{latency:.0f} ms")
    cols[3].metric("Run ID", run_id[:16] + "…" if len(run_id) > 16 else run_id)

    st.caption(f"Full run ID: `{run_id}` — [View trace in Phoenix]({phoenix_endpoint})")


def _render_evidence(result: dict[str, Any]) -> None:
    """Render opened evidence chunks."""
    opened = result.get("opened", [])
    if not opened:
        return

    with st.expander(f"📄 Evidence ({len(opened)} chunks)", expanded=False):
        for chunk in opened:
            title = chunk.get("title", "Untitled")
            section = chunk.get("section", "")
            chunk_id = chunk.get("chunk_id", "?")
            text = chunk.get("text", "")
            header = f"**{title}**"
            if section:
                header += f" › {section}"
            header += f"  `{chunk_id}`"
            st.markdown(header)
            st.text(text[:500] + ("…" if len(text) > 500 else ""))
            st.divider()


def _render_web_results(result: dict[str, Any]) -> None:
    """Render web search results."""
    web_results = result.get("web_results", [])
    if not web_results:
        return

    with st.expander(f"🌐 Web results ({len(web_results)})", expanded=False):
        for r in web_results:
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            st.markdown(f"**[{title}]({url})**")
            st.caption(snippet)


def _render_calculator_result(result: dict[str, Any]) -> None:
    """Render calculator result from the math route."""
    calc = result.get("calculator_result")
    if not calc:
        return

    with st.expander("🔢 Calculator", expanded=False):
        expr = calc.get("expression", "")
        res = calc.get("result")
        err = calc.get("error")
        if res is not None:
            st.code(f"{expr} = {res}", language=None)
        elif err:
            st.warning(f"Expression: {expr}\nError: {err}")


# ------------------------------------------------------------------ #
#  Graph execution with streaming updates                              #
# ------------------------------------------------------------------ #


def _run_graph_streaming(
    graph,
    question: str,
    graph_name: str,
    cfg,
) -> dict[str, Any]:
    """
    Run a compiled LangGraph with .stream() and show node progress.

    Injects chat history from session state into the initial graph state
    so the rewrite_query node can resolve coreferences.

    Returns the final accumulated state dict.
    """
    from workbench.core.config import create_config_snapshot
    from workbench.core.run_context import run_context
    from workbench.observability.logging import get_logger
    from workbench.observability.metrics import get_metrics_store
    from workbench.observability.tracing import setup_tracing, start_span

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

        # Inject conversation context for multi-turn chat
        chat_state = _build_chat_state()
        if chat_state["chat_history"]:
            initial_state["chat_history"] = chat_state["chat_history"]
        if chat_state["conversation_summary"]:
            initial_state["conversation_summary"] = chat_state["conversation_summary"]

        # --- Stream through graph nodes with live UI updates ----------
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
                        # LangGraph .stream() yields dicts like:
                        #   {"node_name": {partial_state_update}}
                        for node_name, update in event.items():
                            # Show human-readable step label
                            label = _NODE_LABELS.get(
                                node_name,
                                f"⚙️ Running {node_name}…",
                            )
                            st.write(label)

                            # Show brief stats for some nodes
                            if (
                                node_name == "rewrite_query"
                                and "rewritten_query" in update
                            ):
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
                                titles = [
                                    d.get("title", "?") for d in update["opened"][:5]
                                ]
                                st.caption(
                                    f"  → Opened {count} chunks: {', '.join(titles)}"
                                )
                            elif node_name == "timeline" and "timeline" in update:
                                count = len(update["timeline"])
                                st.caption(f"  → Extracted {count} timeline events")
                            elif node_name == "web_search" and "web_results" in update:
                                count = len(update["web_results"])
                                st.caption(f"  → Got {count} web results")
                            elif (
                                node_name == "validate"
                                and "citation_validation" in update
                            ):
                                valid = update["citation_validation"].get("valid", "?")
                                st.caption(f"  → Validation passed: {valid}")
                            elif node_name == "repair":
                                st.caption("  → Re-generating with citation fixes…")
                            # --- Supervisor-specific node captions ---
                            elif node_name == "route_question" and "route" in update:
                                route = update["route"]
                                method = update.get("router_method", "?")
                                conf = update.get("router_confidence", 0)
                                st.caption(
                                    f"  → Route: **{route}** "
                                    f"(method={method}, conf={conf:.0%})"
                                )
                            elif (
                                node_name == "run_math"
                                and "calculator_result" in update
                            ):
                                calc = update["calculator_result"]
                                if calc and calc.get("result") is not None:
                                    st.caption(
                                        f"  → {calc['expression']} = {calc['result']}"
                                    )
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

                            # Merge partial update into accumulated state
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

        # --- Logging / metrics ----------------------------------------
        logger.log_event(
            f"{graph_name}_completed",
            {
                "question": question[:200],
                "rewritten_query": accumulated.get("rewritten_query", "")[:200],
                "answer_length": len(accumulated.get("answer_text", "")),
                "citation_count": len(accumulated.get("citations", [])),
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

        # Stash run_id in the result for display
        accumulated["_run_id"] = run_id

        return accumulated


# ------------------------------------------------------------------ #
#  Session state initialisation                                        #
# ------------------------------------------------------------------ #


def _init_session_state() -> None:
    """Initialise session state keys on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph_name" not in st.session_state:
        st.session_state.graph_name = "rag_graph"
    if "conversation_summary" not in st.session_state:
        st.session_state.conversation_summary = ""


# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #


def _render_sidebar(cfg) -> str:
    """Render sidebar controls and return the selected graph name."""
    with st.sidebar:
        st.title("🔬 Agentic Workbench")
        st.caption("Local RAG chat with LangGraph")

        st.divider()

        # --- System selector -----------------------------------------
        graph_options = {
            "rag_graph": "RAG (search → open → answer)",
            "researcher_graph": "Researcher (+ timeline + citation validation)",
            "supervisor_graph": "Supervisor (auto-routes to best specialist)",
        }
        selected = st.selectbox(
            "System",
            options=list(graph_options.keys()),
            format_func=lambda k: graph_options[k],
            index=list(graph_options.keys()).index(st.session_state.graph_name),
            key="graph_selector",
        )
        st.session_state.graph_name = selected

        # --- Routing mode (only for supervisor) -----------------------
        if selected == "supervisor_graph":
            supervisor_cfg = getattr(cfg, "supervisor", {}) or {}
            current_mode = supervisor_cfg.get("routing_mode", "hybrid")
            modes = ["hybrid", "heuristic_only", "llm_only"]
            st.selectbox(
                "Routing mode",
                options=modes,
                index=modes.index(current_mode) if current_mode in modes else 0,
                key="routing_mode_display",
                disabled=True,
                help="Set in configs/defaults.yaml → supervisor.routing_mode",
            )

        st.divider()

        # --- Conversation info ----------------------------------------
        msg_count = len(st.session_state.get("messages", []))
        summary = st.session_state.get("conversation_summary", "")
        st.markdown(f"**Conversation**: {msg_count} messages")
        if summary:
            with st.expander("📝 Summary", expanded=False):
                st.caption(summary)

        st.divider()

        # --- Model info -----------------------------------------------
        st.markdown("**Model**")
        st.caption(
            f"{cfg.model.get('model_name', 'unknown')} @ {cfg.model['base_url']}"
        )

        # --- Service info ---------------------------------------------
        emb_provider = cfg.embeddings.get("provider", "http")
        rr_provider = cfg.reranking.get("provider", "http")
        st.markdown("**Services**")
        if emb_provider == "http":
            st.caption(f"Embedder: {cfg.embeddings.get('base_url', '?')}")
        else:
            st.caption(f"Embedder: local ({cfg.embeddings['model_name']})")
        if rr_provider == "http":
            st.caption(f"Reranker: {cfg.reranking.get('base_url', '?')}")
        else:
            st.caption(f"Reranker: local ({cfg.reranking['model_name']})")

        # --- Phoenix link ---------------------------------------------
        phoenix_url = cfg.observability.get("phoenix_endpoint", "http://localhost:6006")
        st.markdown(f"**Traces**: [Phoenix UI]({phoenix_url})")

        st.divider()

        # --- Clear chat -----------------------------------------------
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_summary = ""
            st.rerun()

        # --- Config info -----------------------------------------------
        with st.expander("⚙️ Config", expanded=False):
            st.json(
                {
                    "retrieval": cfg.retrieval,
                    "reranking": cfg.reranking,
                    "web_search": cfg.web_search,
                    "supervisor": getattr(cfg, "supervisor", {}),
                    "embeddings": {
                        "provider": cfg.embeddings.get("provider", "http"),
                        "model_name": cfg.embeddings["model_name"],
                    },
                },
                expanded=False,
            )

    return selected


# ------------------------------------------------------------------ #
#  Main app                                                            #
# ------------------------------------------------------------------ #


def main() -> None:
    """Entry point for the Streamlit app."""
    _init_session_state()

    # --- Load config + sidebar ----------------------------------------
    try:
        cfg = _load_config()
    except Exception as exc:
        st.error(
            f"Failed to load config: {exc}\n\n"
            "Make sure `configs/defaults.yaml` exists and is valid."
        )
        st.stop()

    graph_name = _render_sidebar(cfg)

    # --- Header -------------------------------------------------------
    st.title("🔬 Agentic Workbench Chat")

    # --- Build components (cached) ------------------------------------
    try:
        model = _build_model(cfg)
        embedder = _build_embedder(cfg)
        reranker = _build_reranker(cfg)
        chunk_store = _build_chunk_store(cfg)
        retriever = _build_retriever(cfg, embedder, reranker, chunk_store)
        tools = _build_tools(cfg, retriever, chunk_store, model)
    except Exception as exc:
        st.error(
            f"Failed to build components: {type(exc).__name__}: {exc}\n\n"
            "Make sure:\n"
            "1. LanceDB indexes are built (`python scripts/build_vector_index.py`)\n"
            "2. Model server is running (`bash scripts/start_model_server.sh`)\n"
            "3. Embedding + reranker servers are running "
            "(`bash scripts/start_model_stack.sh`)\n"
            "4. All dependencies are installed (`pip install -r requirements.txt`)"
        )
        st.stop()

    # --- Build graph (not cached — depends on selected system) --------
    try:
        graph = _build_graph(graph_name, cfg, model, tools)
    except Exception as exc:
        st.error(f"Failed to compile graph '{graph_name}': {exc}")
        st.stop()

    # --- Render chat history ------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render metadata for assistant messages
            if msg["role"] == "assistant" and "result" in msg:
                result = msg["result"]
                _render_citations(result)
                if graph_name in ("researcher_graph", "supervisor_graph"):
                    _render_timeline(result)
                if graph_name == "supervisor_graph":
                    _render_routing_info(result)
                    _render_calculator_result(result)
                _render_evidence(result)
                _render_web_results(result)
                if "_run_id" in result:
                    _render_run_metadata(result, result["_run_id"], cfg)

    # --- Chat input ---------------------------------------------------
    if prompt := st.chat_input("Ask a question…"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the graph and display assistant response
        with st.chat_message("assistant"):
            result = _run_graph_streaming(graph, prompt, graph_name, cfg)

            # Display the answer
            answer = result.get("answer_text", "")
            error = result.get("error")

            if error and not answer:
                st.error(f"Error: {error}")
                answer = f"⚠️ Error: {error}"
            elif error and answer:
                st.warning(f"Completed with warning: {error}")
                st.markdown(answer)
            else:
                st.markdown(answer)

            # Render expandable panels
            _render_citations(result)
            if graph_name in ("researcher_graph", "supervisor_graph"):
                _render_timeline(result)
            if graph_name == "supervisor_graph":
                _render_routing_info(result)
                _render_calculator_result(result)
            _render_evidence(result)
            _render_web_results(result)
            if "_run_id" in result:
                _render_run_metadata(result, result["_run_id"], cfg)

        # Store assistant message + result for history replay
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "result": result,
            }
        )

        # Update conversation summary if needed
        _update_conversation_summary(model, cfg)


# ------------------------------------------------------------------ #
#  Entrypoint                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    main()
