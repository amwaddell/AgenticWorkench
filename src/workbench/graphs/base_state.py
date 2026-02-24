"""
Shared state schemas for LangGraph-based RAG workflows.

All graphs in this package use ``RAGState`` (or a superset of it)
as their state type.  LangGraph nodes receive the full state dict
and return a *partial* dict of fields to update.

State schemas
-------------
``RAGState``          — Day 3: basic search → open → answer
``ResearcherState``   — Day 4: adds timeline, citation validation,
                        and repair-loop tracking

Field categories
----------------
**Input** — set by the caller at ``invoke()`` time:
    question, search_top_k, open_top_n

**Retrieval** — written by ``retrieve_node``:
    retrieved

**Evidence** — written by ``open_node``:
    opened

**Timeline** — written by ``timeline_node`` (researcher only):
    timeline, timeline_raw

**Answer** — written by ``answer_node``:
    answer_text, citations, tokens_in, tokens_out,
    model_latency_ms, prompt_meta

**Validation** — written by ``validate_node`` (researcher only):
    citation_validation, repair_attempts

**Error** — set by any node on failure:
    error

Usage::

    from workbench.graphs.base_state import RAGState, ResearcherState

    # Basic RAG
    result = rag_graph.invoke({"question": "..."})

    # Researcher (superset of RAGState)
    result = researcher_graph.invoke({"question": "..."})
    print(result["timeline"])
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

    # --- Input (set by caller) ------------------------------------
    question: str
    search_top_k: int  # default 5, overridable at invoke
    open_top_n: int  # default 3, overridable at invoke

    # --- Produced by retrieve_node --------------------------------
    retrieved: list[dict[str, Any]]
    # Each dict matches RetrievedChunkSummary.model_dump():
    #   chunk_id, score, title, snippet

    # --- Produced by open_node ------------------------------------
    opened: list[dict[str, Any]]
    # Each dict matches OpenedChunk.model_dump():
    #   chunk_id, document_id, title, section, text

    # --- Produced by answer_node ----------------------------------
    answer_text: str
    citations: list[str]
    tokens_in: int
    tokens_out: int
    model_latency_ms: float
    prompt_meta: dict[str, Any]
    # prompt_meta includes: variant, template, version,
    #   evidence_count, timeline_count

    # --- Error tracking -------------------------------------------
    error: str


class ResearcherState(RAGState, total=False):
    """
    Extended state for the researcher graph.

    Inherits every field from ``RAGState`` and adds timeline,
    citation-validation, and repair-loop fields.
    """

    # --- Produced by timeline_node --------------------------------
    timeline: list[dict[str, Any]]
    # Each dict matches TimelineItem.model_dump():
    #   date, event, supporting_chunk_ids
    timeline_raw: str
    # Raw model output from the timeline extraction call
    #   (useful for debugging / eval)

    # --- Produced by validate_node --------------------------------
    citation_validation: dict[str, Any]
    # Serialised ValidationResult from citations.validate_citations()
    #   keys: valid, errors, warnings, cited_ids, hallucinated_ids,
    #         uncited_opened_ids, paragraph_count,
    #         paragraphs_with_citations

    # --- Repair loop tracking -------------------------------------
    repair_attempts: int
    # How many times the answer has been regenerated to fix citations.
    # The graph caps this at policy.max_repair_attempts.
