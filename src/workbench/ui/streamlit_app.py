"""
Streamlit local chat UI for the agentic workbench.

Features
--------
* ChatGPT-style conversation interface with multi-turn context
* Query rewriting for follow-up questions (resolves "that", "it", etc.)
* System selector: ``rag_graph``, ``researcher_graph``, or ``supervisor_graph``
* Node-level streaming — shows live progress as the graph executes
* Numbered citations ``[1]``, ``[2]``, etc. with article titles and excerpts
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
#  Imports from sub-modules                                            #
# ------------------------------------------------------------------ #

from workbench.ui.chat_state import (  # noqa: E402
    init_session_state,
    update_conversation_summary,
)
from workbench.ui.components import (  # noqa: E402
    build_chunk_store,
    build_embedder,
    build_graph,
    build_model,
    build_reranker,
    build_retriever,
    build_tools,
    load_config,
)
from workbench.ui.graph_runner import run_graph_streaming  # noqa: E402
from workbench.ui.renderers import (  # noqa: E402
    render_calculator_result,
    render_citations,
    render_evidence,
    render_routing_info,
    render_run_metadata,
    render_timeline,
    render_web_results,
)

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
                index=(modes.index(current_mode) if current_mode in modes else 0),
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
#  Render all result panels                                            #
# ------------------------------------------------------------------ #


def _render_result_panels(result: dict[str, Any], graph_name: str, cfg) -> None:
    """Render all expandable panels for a graph result."""
    render_citations(result)
    if graph_name in ("researcher_graph", "supervisor_graph"):
        render_timeline(result)
    if graph_name == "supervisor_graph":
        render_routing_info(result)
        render_calculator_result(result)
    render_evidence(result)
    render_web_results(result)
    if "_run_id" in result:
        render_run_metadata(result, result["_run_id"], cfg)


# ------------------------------------------------------------------ #
#  Main app                                                            #
# ------------------------------------------------------------------ #


def main() -> None:
    """Entry point for the Streamlit app."""
    init_session_state()

    # --- Load config + sidebar ----------------------------------------
    try:
        cfg = load_config()
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
        model = build_model(cfg)
        embedder = build_embedder(cfg)
        reranker = build_reranker(cfg)
        chunk_store = build_chunk_store(cfg)
        retriever = build_retriever(cfg, embedder, reranker, chunk_store)
        tools = build_tools(cfg, retriever, chunk_store, model)
    except Exception as exc:
        st.error(
            f"Failed to build components: {type(exc).__name__}: {exc}\n\n"
            "Make sure:\n"
            "1. LanceDB indexes are built "
            "(`python scripts/build_vector_index.py`)\n"
            "2. Model server is running "
            "(`bash scripts/start_model_server.sh`)\n"
            "3. Embedding + reranker servers are running "
            "(`bash scripts/start_model_stack.sh`)\n"
            "4. All dependencies are installed "
            "(`pip install -r requirements.txt`)"
        )
        st.stop()

    # --- Build graph (not cached — depends on selected system) --------
    try:
        graph = build_graph(graph_name, cfg, model, tools)
    except Exception as exc:
        st.error(f"Failed to compile graph '{graph_name}': {exc}")
        st.stop()

    # --- Render chat history ------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "result" in msg:
                _render_result_panels(msg["result"], graph_name, cfg)

    # --- Chat input ---------------------------------------------------
    if prompt := st.chat_input("Ask a question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the graph and display assistant response
        with st.chat_message("assistant"):
            result = run_graph_streaming(graph, prompt, graph_name, cfg)

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

            _render_result_panels(result, graph_name, cfg)

        # Store assistant message + result for history replay
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "result": result,
            }
        )

        # Update conversation summary if needed
        update_conversation_summary(model, cfg)


# ------------------------------------------------------------------ #
#  Entrypoint                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    main()
