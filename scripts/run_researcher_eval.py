#!/usr/bin/env python3
"""
Run the historical researcher evaluation.

Iterates over the built-in history question set, runs the researcher
*graph* on each question, computes measures, and writes:
    - per-question Markdown reports  → runs/reports/<run_id>/
    - summary report                 → runs/reports/<run_id>/summary.md
    - raw results JSON               → runs/reports/<run_id>/results.json

This script uses the LangGraph-based ``researcher_graph`` (Day 4+).
The legacy ``AgentLoop`` + ``ResearcherPolicy`` path has been retired.

Usage:
    python scripts/run_researcher_eval.py
    python scripts/run_researcher_eval.py --questions 5
    python scripts/run_researcher_eval.py --question-id hist_001
    python scripts/run_researcher_eval.py --system supervisor_graph
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from workbench.agents.state import (
    AgentState,
    OpenedChunk,
    RetrievedChunkSummary,
    TimelineItem,
)
from workbench.core.config import create_config_snapshot, load_config
from workbench.core.run_context import run_context
from workbench.data_build.embeddings import SentenceTransformerEmbedder
from workbench.evaluation.datasets import load_history_questions
from workbench.evaluation.measures import compute_all_measures
from workbench.evaluation.reports import (
    generate_report,
    generate_summary_report,
    save_report,
)
from workbench.graphs.researcher_graph import build_researcher_graph
from workbench.graphs.supervisor_graph import build_supervisor_graph
from workbench.models.llamacpp_server import LlamaCppServerModel
from workbench.observability.logging import get_logger
from workbench.observability.phoenix import setup_phoenix_tracing
from workbench.observability.tracing import start_span
from workbench.prompting.prompt_builder import PromptBuilder
from workbench.retrieval.hybrid import HybridRetriever
from workbench.retrieval.keyword_search import LanceDBKeywordSearcher
from workbench.retrieval.rerankers import CrossEncoderReranker
from workbench.retrieval.vector_search import LanceDBVectorSearcher
from workbench.stores.chunk_store import ChunkStore
from workbench.tools.open_chunk import OpenChunkTool
from workbench.tools.search_wikipedia import SearchWikipediaTool
from workbench.tools.timeline import TimelineTool

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _graph_result_to_agent_state(
    result: dict,
    question: str,
) -> AgentState:
    """
    Convert a graph result dict to an ``AgentState``.

    The evaluation measures and report generators expect ``AgentState``,
    but the LangGraph returns a plain dict (``ResearcherState``).
    This adapter bridges the two.
    """
    state = AgentState(question=question)

    # Retrieval stage
    for r in result.get("retrieved", []):
        if isinstance(r, dict):
            state.retrieved.append(
                RetrievedChunkSummary(
                    chunk_id=r.get("chunk_id", ""),
                    score=r.get("score", 0.0),
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                )
            )

    # Evidence stage
    for o in result.get("opened", []):
        if isinstance(o, dict):
            state.opened.append(
                OpenedChunk(
                    chunk_id=o.get("chunk_id", ""),
                    document_id=o.get("document_id", ""),
                    title=o.get("title", ""),
                    section=o.get("section"),
                    text=o.get("text", ""),
                )
            )

    # Timeline stage
    for t in result.get("timeline", []):
        if isinstance(t, dict):
            state.timeline.append(
                TimelineItem(
                    date=t.get("date", ""),
                    event=t.get("event", ""),
                    supporting_chunk_ids=t.get("supporting_chunk_ids", []),
                )
            )
        elif isinstance(t, TimelineItem):
            state.timeline.append(t)

    # Generation stage
    state.answer_text = result.get("answer_text", "")
    state.citations = result.get("citations", [])
    state.tokens_in = result.get("tokens_in", 0)
    state.tokens_out = result.get("tokens_out", 0)
    state.model_latency_ms = result.get("model_latency_ms", 0.0)
    state.done = True
    state.error = result.get("error")

    return state


# ------------------------------------------------------------------
# Component building
# ------------------------------------------------------------------


def build_components(cfg):
    """Build all pipeline components from config."""
    db_path = Path(cfg.paths["active_index"]) / "lancedb"

    model = LlamaCppServerModel(
        base_url=cfg.model["base_url"],
        model_name=cfg.model.get("model_name", "local-model"),
        timeout_seconds=cfg.model.get("timeout_seconds", 60),
        default_temperature=cfg.model.get("temperature", 0.7),
        default_max_tokens=cfg.model.get("max_tokens", 2048),
    )

    embedder = SentenceTransformerEmbedder(
        model_name=cfg.embeddings["model_name"],
        device=cfg.embeddings.get("device", "mps"),
        batch_size=cfg.embeddings.get("batch_size", 32),
    )

    keyword_searcher = LanceDBKeywordSearcher(db_path=db_path)
    vector_searcher = LanceDBVectorSearcher(db_path=db_path, embedder=embedder)

    reranker = None
    if cfg.reranking.get("enabled", True):
        reranker = CrossEncoderReranker(model_name=cfg.reranking["model_name"])

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

    search_tool = SearchWikipediaTool(retriever=retriever)
    open_chunk_tool = OpenChunkTool(chunk_store=chunk_store)
    prompt_builder = PromptBuilder()
    timeline_tool = TimelineTool(
        model=model,
        prompt_builder=prompt_builder,
        generation_settings={"temperature": 0.3, "max_tokens": 2048},
    )

    return {
        "model": model,
        "search_tool": search_tool,
        "open_chunk_tool": open_chunk_tool,
        "prompt_builder": prompt_builder,
        "timeline_tool": timeline_tool,
    }


# ------------------------------------------------------------------
# Graph building
# ------------------------------------------------------------------


def build_eval_graph(system: str, components: dict, cfg):
    """Build the evaluation graph for the given system name."""
    common = dict(
        search_tool=components["search_tool"],
        open_chunk_tool=components["open_chunk_tool"],
        model=components["model"],
        prompt_builder=components["prompt_builder"],
    )

    if system == "researcher_graph":
        return build_researcher_graph(
            **common,
            timeline_tool=components["timeline_tool"],
            generation_settings={"temperature": 0.5, "max_tokens": 2048},
            open_top_n=5,
            search_top_k=10,
        )

    if system == "supervisor_graph":
        sup_cfg = getattr(cfg, "supervisor", {}) or {}
        return build_supervisor_graph(
            **common,
            timeline_tool=components["timeline_tool"],
            routing_mode=sup_cfg.get("routing_mode", "hybrid"),
            open_top_n=5,
            search_top_k=10,
        )

    raise ValueError(f"Unknown system: {system!r}")


# ------------------------------------------------------------------
# Run a single question
# ------------------------------------------------------------------


def run_single_question(
    question_data: dict,
    graph: object,
    open_top_n: int = 5,
    search_top_k: int = 10,
) -> dict:
    """Run the graph on a single question and return results."""
    question = question_data["question"]
    qid = question_data["id"]

    print(f"\n{'=' * 60}")
    print(f"[{qid}] {question}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Build initial state and invoke the graph
    initial_state = {
        "question": question,
        "search_top_k": search_top_k,
        "open_top_n": open_top_n,
    }
    result = graph.invoke(initial_state)

    total_ms = (time.time() - t0) * 1000

    # Convert graph result → AgentState for measures/reports
    state = _graph_result_to_agent_state(result, question)

    # Compute measures
    metrics = compute_all_measures(state)
    metrics["total_latency_ms"] = round(total_ms, 1)

    # Print summary
    print("  Steps: retrieve → open → timeline → answer → validate")
    print(f"  Citations: {len(state.citations)}")
    print(f"  Timeline items: {len(state.timeline)}")
    print(f"  Total latency: {total_ms:.0f} ms")
    if state.error:
        print(f"  ERROR: {state.error}")

    return {
        "question_id": qid,
        "question": question,
        "question_meta": question_data,
        "state": state,
        "metrics": metrics,
        "error": state.error,
        "total_ms": total_ms,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run researcher evaluation")
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Number of questions to run (default: all 20)",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Run a single question by ID (e.g. hist_001)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="researcher_graph",
        choices=["researcher_graph", "supervisor_graph"],
        help="Which graph system to evaluate (default: researcher_graph)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path=config_path)
    config_snapshot = create_config_snapshot(cfg)

    # Set up tracing (sends spans to Phoenix on localhost:6006)
    setup_phoenix_tracing(service_name="researcher-eval")

    # Load questions
    questions = load_history_questions()

    if args.question_id:
        questions = [q for q in questions if q["id"] == args.question_id]
        if not questions:
            print(f"Question ID '{args.question_id}' not found.")
            sys.exit(1)
    elif args.questions:
        questions = questions[: args.questions]

    print(f"Running evaluation on {len(questions)} questions...")
    print(f"System: {args.system}")

    # Build components once
    print("Building components...")
    components = build_components(cfg)
    print("Components ready.")

    # Build graph once
    print(f"Compiling {args.system}...")
    graph = build_eval_graph(args.system, components, cfg)
    print("Graph compiled.\n")

    # Run within a traced context
    with run_context(config_snapshot, run_type="evaluation") as ctx:
        run_id = ctx.run_id
        logger = get_logger(run_id=run_id)
        logger.log_run_started(run_id, "evaluation", config_snapshot)

        reports_dir = Path(cfg.runs.get("reports_dir", "runs/reports")) / run_id
        reports_dir.mkdir(parents=True, exist_ok=True)

        all_results: list[dict] = []

        with start_span("eval.run", attributes={"question_count": len(questions)}):
            for q in questions:
                try:
                    result = run_single_question(q, graph)
                    all_results.append(result)

                    # Write per-question report
                    md = generate_report(
                        state=result["state"],
                        metrics=result["metrics"],
                        run_id=run_id,
                        question_meta=result["question_meta"],
                    )
                    save_report(md, reports_dir / f"{result['question_id']}.md")

                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    all_results.append(
                        {
                            "question_id": q["id"],
                            "question": q["question"],
                            "metrics": {},
                            "error": str(exc),
                        }
                    )

        # Write summary report
        summary_md = generate_summary_report(all_results, run_id=run_id)
        summary_path = save_report(summary_md, reports_dir / "summary.md")
        print(f"\nSummary report: {summary_path}")

        # Write raw results JSON (without AgentState objects)
        json_results = []
        for r in all_results:
            entry = {
                "question_id": r.get("question_id"),
                "question": r.get("question"),
                "metrics": r.get("metrics", {}),
                "error": r.get("error"),
            }
            if "state" in r:
                entry["answer_text"] = r["state"].answer_text
                entry["citations"] = r["state"].citations
                entry["timeline"] = [t.model_dump() for t in r["state"].timeline]
            json_results.append(entry)

        results_path = reports_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"Raw results:    {results_path}")
        print(f"Run ID:         {run_id}")


if __name__ == "__main__":
    main()
