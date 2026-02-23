"""
Chatbot system: assemble components and run a question end-to-end.

This is the top-level entry point that:
    1. Loads configuration
    2. Builds all components (model, retriever, reranker, tools)
    3. Wires them into an agent loop
    4. Runs a single question within a traced run context
    5. Returns the final agent state (answer + citations + metrics)

Usage:
    from workbench.systems.chatbot import ChatbotSystem

    system = ChatbotSystem()               # uses defaults.yaml
    result = system.ask("When did the Roman Republic end?")
    print(result.answer_text)
    print(result.citations)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from workbench.agents.loops import AgentLoop
from workbench.agents.policies import ScriptedPolicy
from workbench.agents.state import AgentState
from workbench.core.config import (
    WorkbenchConfig,
    create_config_snapshot,
    load_config,
)
from workbench.core.run_context import run_context
from workbench.data_build.embeddings import SentenceTransformerEmbedder
from workbench.models.llamacpp_server import LlamaCppServerModel
from workbench.observability.logging import get_logger
from workbench.observability.metrics import get_metrics_store
from workbench.observability.tracing import setup_tracing, start_span
from workbench.retrieval.hybrid import HybridRetriever
from workbench.retrieval.keyword_search import LanceDBKeywordSearcher
from workbench.retrieval.rerankers import CrossEncoderReranker
from workbench.retrieval.vector_search import LanceDBVectorSearcher
from workbench.stores.chunk_store import ChunkStore
from workbench.tools.open_chunk import OpenChunkTool
from workbench.tools.search_wikipedia import SearchWikipediaTool


class ChatbotSystem:
    """
    Fully assembled chatbot: retrieval + reranking + LLM generation.

    All components are built lazily on first ``ask()`` call so you can
    create the object cheaply and only pay for initialisation when needed.

    Args:
        config_path: Path to YAML config (default: configs/defaults.yaml).
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
        self._components_built = False

        # Component slots (filled by _build_components)
        self._model: LlamaCppServerModel | None = None
        self._embedder: SentenceTransformerEmbedder | None = None
        self._keyword_searcher: LanceDBKeywordSearcher | None = None
        self._vector_searcher: LanceDBVectorSearcher | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._chunk_store: ChunkStore | None = None
        self._retriever: HybridRetriever | None = None
        self._search_tool: SearchWikipediaTool | None = None
        self._open_chunk_tool: OpenChunkTool | None = None

    def _build_components(self) -> None:
        """Build all components from config (called once)."""
        if self._components_built:
            return

        cfg = self.config
        db_path = Path(cfg.paths["active_index"]) / "lancedb"

        # 1. Model
        self._model = LlamaCppServerModel(
            base_url=cfg.model["base_url"],
            model_name=cfg.model.get("model_name", "local-model"),
            timeout_seconds=cfg.model.get("timeout_seconds", 60),
            default_temperature=cfg.model.get("temperature", 0.7),
            default_max_tokens=cfg.model.get("max_tokens", 2048),
        )

        # 2. Embedder
        self._embedder = SentenceTransformerEmbedder(
            model_name=cfg.embeddings["model_name"],
            device=cfg.embeddings.get("device", "mps"),
            batch_size=cfg.embeddings.get("batch_size", 32),
        )

        # 3. Searchers
        self._keyword_searcher = LanceDBKeywordSearcher(db_path=db_path)
        self._vector_searcher = LanceDBVectorSearcher(
            db_path=db_path,
            embedder=self._embedder,
        )

        # 4. Reranker (optional)
        self._reranker = None
        if cfg.reranking.get("enabled", True):
            self._reranker = CrossEncoderReranker(
                model_name=cfg.reranking["model_name"],
            )

        # 5. Chunk store — single source of truth for chunk lookups
        self._chunk_store = ChunkStore(db_path=db_path)

        # 6. Hybrid retriever
        retrieval_cfg = cfg.retrieval
        self._retriever = HybridRetriever(
            keyword_searcher=self._keyword_searcher,
            vector_searcher=self._vector_searcher,
            reranker=self._reranker,
            chunk_store=self._chunk_store,
            keyword_top_k=retrieval_cfg.get("keyword_top_k", 20),
            vector_top_k=retrieval_cfg.get("vector_top_k", 20),
            merge_top_n=retrieval_cfg.get("merge_top_n", 50),
            rrf_k=retrieval_cfg.get("rrf_k", 60),
        )

        # 7. Tools
        self._search_tool = SearchWikipediaTool(retriever=self._retriever)
        self._open_chunk_tool = OpenChunkTool(chunk_store=self._chunk_store)

        self._components_built = True
        print("Components built successfully.")

    def ask(
        self,
        question: str,
        open_top_n: int = 3,
        search_top_k: int = 5,
        max_steps: int = 10,
    ) -> AgentState:
        """
        Ask a question and get an answer with citations.

        Args:
            question: The user's question.
            open_top_n: How many retrieved chunks to open.
            search_top_k: How many chunks to retrieve.
            max_steps: Maximum agent loop iterations.

        Returns:
            Final AgentState with answer_text, citations, and metrics.
        """
        setup_tracing(service_name="agentic-workbench-chatbot")
        self._build_components()

        config_snapshot = create_config_snapshot(self.config)

        with run_context(config_snapshot, run_type="chatbot") as ctx:
            logger = get_logger(run_id=ctx.run_id)
            metrics = get_metrics_store()

            logger.log_run_started(ctx.run_id, "chatbot", config_snapshot)
            logger.log_query(question)

            with start_span(
                "system.chatbot.ask",
                attributes={"question": question[:200]},
            ):
                t0 = time.time()

                # Build policy + loop
                policy = ScriptedPolicy(
                    open_top_n=open_top_n,
                    search_top_k=search_top_k,
                )
                loop = AgentLoop(
                    policy=policy,
                    search_tool=self._search_tool,
                    open_chunk_tool=self._open_chunk_tool,
                    model=self._model,
                    max_steps=max_steps,
                )

                # Run the agent
                state = AgentState(question=question)
                state = loop.run(state)

                total_ms = (time.time() - t0) * 1000

            # --- Log final state ---
            logger.log_event("agent_completed", state.summary())

            if state.retrieved:
                logger.log_retrieval(
                    query=question,
                    num_results=len(state.retrieved),
                    retrieval_method="hybrid",
                    duration_ms=total_ms,
                )

            if state.tokens_out > 0:
                logger.log_model_call(
                    model_name=self.config.model.get("model_name", "unknown"),
                    tokens_in=state.tokens_in,
                    tokens_out=state.tokens_out,
                    duration_ms=state.model_latency_ms,
                )

            # --- Record metrics ---
            metrics.record_component_latency(
                component_name="chatbot.ask",
                duration_ms=total_ms,
                component_type="system",
            )

            if state.tokens_out > 0:
                metrics.record_model_tokens(
                    model_name=self.config.model.get("model_name", "unknown"),
                    tokens_in=state.tokens_in,
                    tokens_out=state.tokens_out,
                    duration_ms=state.model_latency_ms,
                )

            metrics.record_retrieval(
                final_count=len(state.opened),
                query=question,
                keyword_candidates=len(state.retrieved),
                vector_candidates=len(state.retrieved),
                reranked=len(state.retrieved),
                duration_ms=total_ms,
            )

            # Attach run_id to state metadata for caller
            state.question = question  # already set, but be explicit

            print(f"\nRun ID: {ctx.run_id}")
            return state
