"""
Unified graph runner: run any workbench graph with full observability.

This is the recommended entry point for running graphs.  It:
    1. Builds all components from config
    2. Builds the requested graph
    3. Runs it within a traced ``run_context``
    4. Logs, traces, and records metrics
    5. Returns the final graph state dict

Supported systems:
    ``rag_graph``          — search → open → answer
    ``researcher_graph``   — search → open → timeline → answer → validate
    ``supervisor_graph``   — routes to specialist subgraphs

Usage::

    from workbench.systems.runner import GraphRunner

    runner = GraphRunner()                        # uses defaults.yaml
    result = runner.run("researcher_graph", "When did the Roman Republic end?")
    print(result["answer_text"])

Usage (multi-turn chat)::

    runner = GraphRunner()
    result = runner.run(
        "rag_graph",
        "What was that battle about?",
        chat_history=[
            {"role": "user", "content": "Tell me about WW2"},
            {"role": "assistant", "content": "World War II was..."},
        ],
        conversation_summary="Discussing WW2 and the Eastern Front.",
    )

Or as a one-liner::

    from workbench.systems.runner import run_graph
    result = run_graph("rag_graph", "What is photosynthesis?")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from workbench.core.config import (
    WorkbenchConfig,
    create_config_snapshot,
    load_config,
)
from workbench.core.run_context import run_context
from workbench.graphs.rag_graph import build_rag_graph
from workbench.graphs.researcher_graph import build_researcher_graph
from workbench.graphs.supervisor_graph import build_supervisor_graph
from workbench.models.llamacpp_server import LlamaCppServerModel
from workbench.observability.logging import get_logger
from workbench.observability.metrics import get_metrics_store
from workbench.observability.tracing import setup_tracing, start_span
from workbench.prompting.prompt_builder import PromptBuilder
from workbench.retrieval.hybrid import HybridRetriever
from workbench.retrieval.keyword_search import LanceDBKeywordSearcher
from workbench.retrieval.vector_search import LanceDBVectorSearcher
from workbench.stores.chunk_store import ChunkStore
from workbench.tools.open_chunk import OpenChunkTool
from workbench.tools.search_wikipedia import SearchWikipediaTool
from workbench.tools.timeline import TimelineTool

# Graphs that require only search + open + model
_BASIC_GRAPHS = {"rag_graph"}
# Graphs that also need a timeline tool
_TIMELINE_GRAPHS = {"researcher_graph"}
# Graphs that want every tool available
_FULL_GRAPHS = {"supervisor_graph"}

SUPPORTED_GRAPHS = _BASIC_GRAPHS | _TIMELINE_GRAPHS | _FULL_GRAPHS


# ------------------------------------------------------------------ #
#  Factory functions: build embedder / reranker from config            #
# ------------------------------------------------------------------ #


def build_embedder(cfg: WorkbenchConfig) -> Any:
    """
    Build an embedder based on config ``embeddings.provider``.

    - ``"http"``  → lightweight :class:`HTTPEmbedder` (calls remote server)
    - ``"local"`` or ``"sentence_transformers"`` → heavyweight
      :class:`SentenceTransformerEmbedder` (loads model in-process)

    Args:
        cfg: Validated workbench config.

    Returns:
        Object satisfying the ``Embedder`` protocol.
    """
    emb_cfg = cfg.embeddings
    provider = emb_cfg.get("provider", "http")

    if provider == "http":
        from workbench.data_build.http_embedder import HTTPEmbedder

        return HTTPEmbedder(
            base_url=emb_cfg.get("base_url", "http://127.0.0.1:8081"),
            model_name=emb_cfg["model_name"],
            timeout_seconds=emb_cfg.get("timeout_seconds", 120),
        )

    # Local fallback
    from workbench.data_build.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(
        model_name=emb_cfg["model_name"],
        device=emb_cfg.get("device", "mps"),
        batch_size=emb_cfg.get("batch_size", 32),
    )


def build_reranker(cfg: WorkbenchConfig) -> Any | None:
    """
    Build a reranker based on config ``reranking.provider``.

    - ``"http"``           → lightweight :class:`HTTPReranker`
    - ``"cross_encoder"``  → heavyweight :class:`CrossEncoderReranker`
    - disabled             → ``None``

    Args:
        cfg: Validated workbench config.

    Returns:
        Object satisfying the ``Reranker`` protocol, or None.
    """
    rr_cfg = cfg.reranking
    if not rr_cfg.get("enabled", True):
        return None

    provider = rr_cfg.get("provider", "http")

    if provider == "http":
        from workbench.retrieval.http_reranker import HTTPReranker

        return HTTPReranker(
            base_url=rr_cfg.get("base_url", "http://127.0.0.1:8082"),
            model_name=rr_cfg["model_name"],
            timeout_seconds=rr_cfg.get("timeout_seconds", 60),
        )

    # Local fallback
    from workbench.retrieval.rerankers import CrossEncoderReranker

    return CrossEncoderReranker(
        model_name=rr_cfg["model_name"],
    )


class GraphRunner:
    """
    Build components once, run any graph repeatedly.

    Components are built lazily on the first ``run()`` call.

    Args:
        config_path: Path to YAML config (default: ``configs/defaults.yaml``).
        config_overrides: Dict of overrides merged on top of the file.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.config: WorkbenchConfig = load_config(
            config_path=config_path,
            overrides=config_overrides,
        )
        self._components: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    #  Component building                                                 #
    # ------------------------------------------------------------------ #

    def _build_components(self) -> dict[str, Any]:
        """Build all pipeline components from config (called once)."""
        if self._components is not None:
            return self._components

        cfg = self.config
        db_path = Path(cfg.paths["active_index"]) / "lancedb"

        model = LlamaCppServerModel(
            base_url=cfg.model["base_url"],
            model_name=cfg.model.get("model_name", "local-model"),
            timeout_seconds=cfg.model.get("timeout_seconds", 60),
            default_temperature=cfg.model.get("temperature", 0.7),
            default_max_tokens=cfg.model.get("max_tokens", 2048),
        )

        # --- Embedder and reranker via factory functions ---------------
        embedder = build_embedder(cfg)
        reranker = build_reranker(cfg)

        keyword_searcher = LanceDBKeywordSearcher(db_path=db_path)
        vector_searcher = LanceDBVectorSearcher(
            db_path=db_path,
            embedder=embedder,
        )

        chunk_store = ChunkStore(db_path=db_path)

        retrieval_cfg = cfg.retrieval
        retriever = HybridRetriever(
            keyword_searcher=keyword_searcher,
            vector_searcher=vector_searcher,
            reranker=reranker,
            chunk_store=chunk_store,
            keyword_top_k=retrieval_cfg.get("keyword_top_k", 20),
            vector_top_k=retrieval_cfg.get("vector_top_k", 20),
            merge_top_n=retrieval_cfg.get("merge_top_n", 50),
            rrf_k=retrieval_cfg.get("rrf_k", 60),
        )

        prompt_builder = PromptBuilder()
        search_tool = SearchWikipediaTool(retriever=retriever)
        open_chunk_tool = OpenChunkTool(chunk_store=chunk_store)
        timeline_tool = TimelineTool(
            model=model,
            prompt_builder=prompt_builder,
            generation_settings={"temperature": 0.3, "max_tokens": 2048},
        )

        # Optional Day-5 tools (imported lazily to avoid hard failures)
        web_search_tool = None
        calculator_tool = None
        try:
            from workbench.tools.web_search import WebSearchTool

            ws_cfg = getattr(cfg, "web_search", {}) or {}
            if ws_cfg.get("enabled", True):
                web_search_tool = WebSearchTool(
                    provider=ws_cfg.get("provider", "duckduckgo"),
                )
        except Exception:
            pass

        try:
            from workbench.tools.utility.calculator import CalculatorTool

            calculator_tool = CalculatorTool()
        except Exception:
            pass

        self._components = {
            "model": model,
            "search_tool": search_tool,
            "open_chunk_tool": open_chunk_tool,
            "prompt_builder": prompt_builder,
            "timeline_tool": timeline_tool,
            "web_search_tool": web_search_tool,
            "calculator_tool": calculator_tool,
            "chunk_store": chunk_store,
            "config": cfg,
        }
        return self._components

    # ------------------------------------------------------------------ #
    #  Graph building                                                     #
    # ------------------------------------------------------------------ #

    def build_graph(
        self,
        graph_name: str,
        **overrides: Any,
    ) -> Any:
        """
        Build and compile a named graph.

        Args:
            graph_name: One of ``rag_graph``, ``researcher_graph``,
                        ``supervisor_graph``.
            **overrides: Per-graph overrides (``open_top_n``, etc.).

        Returns:
            Compiled LangGraph ready for ``.invoke()`` or ``.stream()``.
        """
        if graph_name not in SUPPORTED_GRAPHS:
            raise ValueError(
                f"Unknown graph {graph_name!r}. Supported: {sorted(SUPPORTED_GRAPHS)}"
            )

        c = self._build_components()
        cfg = c["config"]
        retrieval_cfg = cfg.retrieval

        common = dict(
            search_tool=c["search_tool"],
            open_chunk_tool=c["open_chunk_tool"],
            model=c["model"],
            prompt_builder=c["prompt_builder"],
            open_top_n=overrides.get("open_top_n", retrieval_cfg.get("final_top_k", 5)),
            search_top_k=overrides.get(
                "search_top_k", retrieval_cfg.get("vector_top_k", 10)
            ),
        )

        if graph_name == "rag_graph":
            return build_rag_graph(
                **common,
                web_search_tool=c.get("web_search_tool"),
                confidence_threshold=overrides.get(
                    "confidence_threshold",
                    getattr(cfg, "web_search", {}).get("confidence_threshold", 2),
                ),
            )

        if graph_name == "researcher_graph":
            return build_researcher_graph(
                **common,
                timeline_tool=c["timeline_tool"],
                generation_settings=overrides.get("generation_settings"),
                citation_policy=overrides.get("citation_policy"),
            )

        if graph_name == "supervisor_graph":
            sup_cfg = getattr(cfg, "supervisor", {}) or {}
            return build_supervisor_graph(
                **common,
                timeline_tool=c["timeline_tool"],
                web_search_tool=c.get("web_search_tool"),
                calculator_tool=c.get("calculator_tool"),
                routing_mode=overrides.get(
                    "routing_mode",
                    sup_cfg.get("routing_mode", "hybrid"),
                ),
            )

        raise ValueError(f"Unhandled graph: {graph_name!r}")  # pragma: no cover

    # ------------------------------------------------------------------ #
    #  Run                                                                #
    # ------------------------------------------------------------------ #

    def run(
        self,
        graph_name: str,
        question: str,
        *,
        search_top_k: int | None = None,
        open_top_n: int | None = None,
        chat_history: list[dict[str, str]] | None = None,
        conversation_summary: str | None = None,
        run_type: str | None = None,
        **graph_overrides: Any,
    ) -> dict[str, Any]:
        """
        Build a graph, invoke it, and return the result with full observability.

        Args:
            graph_name: Which graph to run.
            question: The user's question.
            search_top_k: Override default retrieval count.
            open_top_n: Override default open count.
            chat_history: Recent conversation turns for multi-turn chat.
                          Each turn is ``{"role": "user"|"assistant",
                          "content": "..."}``.
            conversation_summary: Rolling summary of older conversation
                                  turns (used alongside chat_history).
            run_type: Label for the run context (default: ``graph_name``).
            **graph_overrides: Forwarded to ``build_graph()``.

        Returns:
            Final graph state dict.
        """
        run_type = run_type or graph_name
        service_name = f"agentic-workbench-{graph_name}"
        config_snapshot = create_config_snapshot(self.config)

        setup_tracing(service_name=service_name)
        graph = self.build_graph(graph_name, **graph_overrides)

        with run_context(config_snapshot, run_type=run_type) as ctx:
            logger = get_logger(run_id=ctx.run_id)
            metrics = get_metrics_store()

            logger.log_run_started(ctx.run_id, run_type, config_snapshot)
            logger.log_query(question)

            with start_span(
                f"{graph_name}.run",
                attributes={"question": question[:200]},
            ):
                t0 = time.time()

                initial_state: dict[str, Any] = {"question": question}
                if search_top_k is not None:
                    initial_state["search_top_k"] = search_top_k
                if open_top_n is not None:
                    initial_state["open_top_n"] = open_top_n
                if chat_history is not None:
                    initial_state["chat_history"] = chat_history
                if conversation_summary is not None:
                    initial_state["conversation_summary"] = conversation_summary

                result: dict[str, Any] = graph.invoke(initial_state)
                total_ms = (time.time() - t0) * 1000

            # --- Logging ---
            logger.log_event(
                f"{graph_name}_completed",
                {
                    "question": question[:200],
                    "rewritten_query": result.get("rewritten_query", "")[:200],
                    "answer_length": len(result.get("answer_text", "")),
                    "citation_count": len(result.get("citations", [])),
                    "retrieved_count": len(result.get("retrieved", [])),
                    "opened_count": len(result.get("opened", [])),
                    "timeline_count": len(result.get("timeline", [])),
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
            model_name = self.config.model.get("model_name", "local-model")
            if tokens_out > 0:
                logger.log_model_call(
                    model_name=model_name,
                    tokens_in=result.get("tokens_in", 0),
                    tokens_out=tokens_out,
                    duration_ms=result.get("model_latency_ms", 0.0),
                )

            # --- Metrics ---
            metrics.record_component_latency(
                component_name=f"{graph_name}.run",
                duration_ms=total_ms,
                component_type="graph",
            )

            if tokens_out > 0:
                metrics.record_model_tokens(
                    model_name=model_name,
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


# ------------------------------------------------------------------ #
#  Module-level convenience                                           #
# ------------------------------------------------------------------ #


def run_graph(
    graph_name: str,
    question: str,
    *,
    config_path: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    chat_history: list[dict[str, str]] | None = None,
    conversation_summary: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    One-shot convenience: build a runner, run a question, return result.

    Args:
        graph_name: Which graph to run.
        question: The user's question.
        config_path: Optional config file path.
        config_overrides: Optional config overrides.
        chat_history: Optional conversation history for multi-turn.
        conversation_summary: Optional rolling summary.
        **kwargs: Forwarded to ``GraphRunner.run()``.

    Returns:
        Final graph state dict.
    """
    runner = GraphRunner(
        config_path=config_path,
        config_overrides=config_overrides,
    )
    return runner.run(
        graph_name,
        question,
        chat_history=chat_history,
        conversation_summary=conversation_summary,
        **kwargs,
    )
