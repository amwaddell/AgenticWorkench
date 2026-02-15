"""
Evaluation report generation.

Produces a Markdown report for a single evaluation run containing:
    - question and answer
    - timeline (if present)
    - sources used (titles + chunk ids)
    - latency and metrics table

Usage:
    from workbench.evaluation.reports import generate_report, save_report
    md = generate_report(state, metrics, run_id="abc123")
    save_report(md, Path("runs/reports/abc123.md"))
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from workbench.agents.state import AgentState


def generate_report(
    state: AgentState,
    metrics: dict[str, Any],
    run_id: str = "unknown",
    question_meta: dict[str, Any] | None = None,
) -> str:
    """
    Generate a Markdown evaluation report for a single question.

    Args:
        state: Completed AgentState.
        metrics: Dict of computed measures (from ``compute_all_measures``).
        run_id: Run identifier.
        question_meta: Optional extra metadata about the question
                       (e.g. topic, expected_date_range).

    Returns:
        Markdown string.
    """
    lines: list[str] = []

    # Header
    lines.append(f"# Evaluation Report: {run_id}")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append("")

    # Question metadata
    lines.append("## Question")
    lines.append(f"**Q:** {state.question}")
    if question_meta:
        if "topic" in question_meta:
            lines.append(f"**Topic:** {question_meta['topic']}")
        if "expected_date_range" in question_meta:
            lines.append(
                f"**Expected date range:** {question_meta['expected_date_range']}"
            )
    lines.append("")

    # Answer
    lines.append("## Answer")
    if state.answer_text:
        lines.append(state.answer_text)
    else:
        lines.append("*No answer generated.*")
    lines.append("")

    # Citations
    lines.append("## Citations")
    if state.citations:
        for cid in state.citations:
            # Find the matching opened chunk for context
            title = ""
            for oc in state.opened:
                if oc.chunk_id == cid:
                    title = oc.title
                    break
            lines.append(f"- `{cid}` — {title}")
    else:
        lines.append("*No citations found in answer.*")
    lines.append("")

    # Timeline
    lines.append("## Timeline")
    if state.timeline:
        lines.append("")
        lines.append("| Date | Event | Sources |")
        lines.append("|------|-------|---------|")
        for item in state.timeline:
            sources = ", ".join(f"`{c}`" for c in item.supporting_chunk_ids)
            # Escape pipe chars in event text
            event = item.event.replace("|", "\\|")
            lines.append(f"| {item.date} | {event} | {sources} |")
    else:
        lines.append("*No timeline extracted.*")
    lines.append("")

    # Sources used
    lines.append("## Sources Used")
    if state.opened:
        seen_titles: set[str] = set()
        for oc in state.opened:
            if oc.title and oc.title not in seen_titles:
                seen_titles.add(oc.title)
                chunk_ids_for_title = [
                    c.chunk_id for c in state.opened if c.title == oc.title
                ]
                ids_str = ", ".join(f"`{c}`" for c in chunk_ids_for_title)
                lines.append(f"- **{oc.title}**: {ids_str}")
    else:
        lines.append("*No sources opened.*")
    lines.append("")

    # Metrics table
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    # Select the most informative metrics for the table
    display_keys = [
        ("citation_count", "Citations in answer"),
        ("unique_cited_titles", "Unique sources cited"),
        ("unique_titles", "Unique sources opened"),
        ("timeline_item_count", "Timeline events"),
        ("avg_citations_per_event", "Avg citations per event"),
        ("retrieved_count", "Chunks retrieved"),
        ("opened_count", "Chunks opened"),
        ("total_steps", "Agent steps"),
        ("model_latency_ms", "Model latency (ms)"),
    ]

    for key, label in display_keys:
        value = metrics.get(key, "—")
        if isinstance(value, float):
            value = f"{value:.1f}"
        lines.append(f"| {label} | {value} |")
    lines.append("")

    # Error (if any)
    if state.error:
        lines.append("## Error")
        lines.append(f"```\n{state.error}\n```")
        lines.append("")

    return "\n".join(lines)


def generate_summary_report(
    results: list[dict[str, Any]],
    run_id: str = "unknown",
) -> str:
    """
    Generate a summary Markdown report across multiple questions.

    Args:
        results: List of dicts, each with ``question``, ``metrics``,
                 and optionally ``error``.
        run_id: Run identifier.

    Returns:
        Markdown string.
    """
    lines: list[str] = []

    lines.append(f"# Evaluation Summary: {run_id}")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append(f"Questions evaluated: {len(results)}")
    lines.append("")

    # Summary table
    lines.append("## Results")
    lines.append("")
    lines.append("| # | Question | Citations | Sources | Timeline | Steps | Error |")
    lines.append("|---|----------|-----------|---------|----------|-------|-------|")

    for i, r in enumerate(results, 1):
        q = r.get("question", "")[:60]
        m = r.get("metrics", {})
        err = "Yes" if r.get("error") else "—"
        lines.append(
            f"| {i} "
            f"| {q} "
            f"| {m.get('citation_count', '—')} "
            f"| {m.get('unique_cited_titles', '—')} "
            f"| {m.get('timeline_item_count', '—')} "
            f"| {m.get('total_steps', '—')} "
            f"| {err} |"
        )

    lines.append("")

    # Averages
    if results:
        avg_citations = _safe_avg(
            [r["metrics"].get("citation_count", 0) for r in results if "metrics" in r]
        )
        avg_sources = _safe_avg(
            [
                r["metrics"].get("unique_cited_titles", 0)
                for r in results
                if "metrics" in r
            ]
        )
        avg_timeline = _safe_avg(
            [
                r["metrics"].get("timeline_item_count", 0)
                for r in results
                if "metrics" in r
            ]
        )
        error_count = sum(1 for r in results if r.get("error"))

        lines.append("## Averages")
        lines.append("")
        lines.append(f"- Avg citations per question: **{avg_citations:.1f}**")
        lines.append(f"- Avg unique sources cited: **{avg_sources:.1f}**")
        lines.append(f"- Avg timeline events: **{avg_timeline:.1f}**")
        lines.append(f"- Errors: **{error_count}** / {len(results)}")
        lines.append("")

    return "\n".join(lines)


def save_report(markdown: str, path: Path) -> Path:
    """
    Save a Markdown report to disk.

    Args:
        markdown: Report content.
        path: Output file path.

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def _safe_avg(values: list[float | int]) -> float:
    """Average that handles empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)
