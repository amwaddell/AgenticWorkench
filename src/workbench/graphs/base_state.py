"""
Shared state schemas for LangGraph-based RAG workflows.

All graphs in this package use ``RAGState`` (or a superset of it)
as their state type.  LangGraph nodes receive the full state dict
and return a *partial* dict of fields to update.

State schemas
-------------
``RAGState``
    Day 3: basic search → open → answer.
    Day 5: adds optional web search fallback fields.

``ResearcherState``
    Day 4: extends RAGState with timeline extraction,
    citation validation, and a repair loop.

``SupervisorState``
    Day 7: extends RAGState with routing metadata and
    subgraph result aggregation for the supervisor graph.

Both use ``total=False`` so callers only need to provide ``question``
at invoke time — the rest is populated by graph nodes.
"""

from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    """
    State schema for the basic RAG graph.

    ``total=False`` means every field is optional in the TypedDict sense,
    which lets callers pass only ``question`` at invoke time.  The graph
    topology guarantees downstream nodes see the fields they need
    (retrieve runs before open, open runs before answer).
    """

    # --- Input --------------------------------------------------------
    question: str

    # --- Retrieval control (overridable per-invoke) -------------------
    search_top_k: int
    open_top_n: int

    # --- Retrieval output ---------------------------------------------
    retrieved: list[dict[str, Any]]  # chunk summaries from search
    opened: list[dict[str, Any]]  # full chunk data after open

    # --- Web search fallback (Day 5) ----------------------------------
    web_search_used: bool  # True if web search was triggered
    web_results: list[dict[str, Any]]  # results from web search

    # --- Answer -------------------------------------------------------
    answer_text: str
    citations: list[str]  # chunk_ids cited in the answer
    web_citations: list[str]  # URLs cited from web search results

    # --- Metadata -----------------------------------------------------
    tokens_in: int
    tokens_out: int
    model_latency_ms: float
    prompt_meta: dict[str, Any]  # template version, variant, etc.

    # --- Error --------------------------------------------------------
    error: str | None


class ResearcherState(RAGState):
    """
    Extended state for the researcher graph.

    Inherits every field from ``RAGState`` and adds timeline,
    citation-validation, and repair-loop fields.
    """

    # --- Timeline (Day 4) --------------------------------------------
    timeline: list[dict[str, Any]]  # parsed timeline items
    timeline_raw: str  # raw model output for debugging

    # --- Citation validation (Day 4) ---------------------------------
    citation_validation: dict[str, Any]  # ValidationResult.to_dict()
    repair_attempts: int


class SupervisorState(RAGState):
    """
    Extended state for the supervisor (hierarchical) graph.

    Inherits every field from ``RAGState`` and adds routing metadata
    and subgraph-specific result fields.  The supervisor routes
    the question to one of several specialist subgraphs, then
    merges the subgraph result back into this unified state.
    """

    # --- Routing (Day 7) ---------------------------------------------
    route: str  # chosen route label: "local_rag", "web_research", etc.
    router_method: str  # "heuristic" or "llm"
    router_confidence: float  # 0.0–1.0 confidence in route choice
    router_rationale: str  # short reason (for logs / debugging)
    router_latency_ms: float  # time spent in the router node

    # --- Subgraph results (Day 7) ------------------------------------
    # Timeline fields (populated when researcher subgraph is used)
    timeline: list[dict[str, Any]]
    timeline_raw: str
    citation_validation: dict[str, Any]
    repair_attempts: int

    # Calculator result (populated when math route is used)
    calculator_result: dict[str, Any]

    # Subgraph provenance
    subgraph_used: str  # name of the subgraph that actually ran
