"""
Evaluation measures: quantify answer quality per question.

Each measure function takes an ``AgentState`` (the completed agent run)
and returns a dict of metric name → value.

Usage:
    from workbench.evaluation.measures import compute_all_measures
    metrics = compute_all_measures(final_state)
    print(metrics)
"""

from __future__ import annotations

from typing import Any

from workbench.agents.state import AgentState


def citation_count(state: AgentState) -> dict[str, Any]:
    """
    Count citations in the answer.

    Returns:
        citation_count: number of unique chunk_ids cited
        citations: the list of cited chunk_ids
    """
    return {
        "citation_count": len(state.citations),
        "citations": state.citations,
    }


def source_diversity(state: AgentState) -> dict[str, Any]:
    """
    Measure how many distinct source documents (by title) were cited.

    A higher number means the answer draws on more independent sources.

    Returns:
        unique_titles: number of distinct document titles in opened chunks
        unique_cited_titles: number of distinct titles for cited chunks only
        title_list: list of unique titles
    """
    # All opened titles
    all_titles = {c.title for c in state.opened if c.title}

    # Titles of cited chunks only
    cited_set = set(state.citations)
    cited_titles = {
        c.title for c in state.opened if c.chunk_id in cited_set and c.title
    }

    return {
        "unique_titles": len(all_titles),
        "unique_cited_titles": len(cited_titles),
        "title_list": sorted(all_titles),
    }


def latency_breakdown(state: AgentState) -> dict[str, Any]:
    """
    Report latency metrics.

    Returns:
        model_latency_ms: time spent in the LLM
        total_steps: number of agent steps
    """
    return {
        "model_latency_ms": state.model_latency_ms,
        "total_steps": state.step,
    }


def tool_call_counts(state: AgentState) -> dict[str, Any]:
    """
    Count how many tool invocations happened.

    Returns:
        retrieved_count: chunks returned by search
        opened_count: chunks the agent opened
        timeline_count: timeline items extracted
    """
    return {
        "retrieved_count": len(state.retrieved),
        "opened_count": len(state.opened),
        "timeline_count": len(state.timeline),
    }


def timeline_quality(state: AgentState) -> dict[str, Any]:
    """
    Measure timeline quality.

    Returns:
        timeline_item_count: number of timeline events
        timeline_cited_chunks: total chunk citations across timeline
        avg_citations_per_event: average citations per timeline event
    """
    if not state.timeline:
        return {
            "timeline_item_count": 0,
            "timeline_cited_chunks": 0,
            "avg_citations_per_event": 0.0,
        }

    total_citations = sum(len(item.supporting_chunk_ids) for item in state.timeline)

    return {
        "timeline_item_count": len(state.timeline),
        "timeline_cited_chunks": total_citations,
        "avg_citations_per_event": round(total_citations / len(state.timeline), 2),
    }


def compute_all_measures(state: AgentState) -> dict[str, Any]:
    """
    Run all measures and return a combined metrics dict.

    Args:
        state: Completed AgentState.

    Returns:
        Flat dict of all metric names → values.
    """
    combined: dict[str, Any] = {}
    combined.update(citation_count(state))
    combined.update(source_diversity(state))
    combined.update(latency_breakdown(state))
    combined.update(tool_call_counts(state))
    combined.update(timeline_quality(state))
    return combined
