"""
Cached component builders for the Streamlit UI.

All heavy objects (model, embedder, retriever, chunk store, tools)
are built here using ``@st.cache_resource`` so they survive across
Streamlit re-runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

# ------------------------------------------------------------------ #
#  Cached builders                                                     #
# ------------------------------------------------------------------ #


@st.cache_resource(show_spinner="Loading configuration…")
def load_config():
    """Load workbench config once."""
    from workbench.core.config import load_config as _load

    return _load()


@st.cache_resource(show_spinner="Loading language model client…")
def build_model(_cfg):
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
def build_embedder(_cfg):
    """
    Build embedder from config.

    With ``provider: http`` this is instant (just creates an HTTP client).
    With ``provider: local`` this loads weights in-process (slow).
    """
    from workbench.systems.runner import build_embedder as _build

    return _build(_cfg)


@st.cache_resource(show_spinner="Connecting to reranker service…")
def build_reranker(_cfg):
    """
    Build reranker from config (or None if disabled).

    With ``provider: http`` this is instant (just creates an HTTP client).
    With ``provider: cross_encoder`` this loads weights in-process (slow).
    """
    from workbench.systems.runner import build_reranker as _build

    return _build(_cfg)


@st.cache_resource(show_spinner="Opening chunk store…")
def build_chunk_store(_cfg):
    """Build the ChunkStore (lazy LanceDB table load)."""
    from workbench.stores.chunk_store import ChunkStore

    db_path = Path(_cfg.paths["active_index"]) / "lancedb"
    return ChunkStore(db_path=db_path)


@st.cache_resource(show_spinner="Building retrieval pipeline…")
def build_retriever(_cfg, _embedder, _reranker, _chunk_store):
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
def build_tools(_cfg, _retriever, _chunk_store, _model):
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

    # Calculator tool
    try:
        from workbench.tools.utility.calculator import CalculatorTool

        tools["calculator"] = CalculatorTool()
    except Exception:
        pass

    return tools


def build_graph(graph_name: str, cfg, model, tools):
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
