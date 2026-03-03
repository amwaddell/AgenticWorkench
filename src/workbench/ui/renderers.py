"""
Rendering helpers for the Streamlit chat UI.

Each ``render_*`` function takes a graph result dict and renders
an expandable panel below the answer.  The citation renderer uses
the ``citation_map`` produced by ``numberize_result()`` to display
clean numbered references instead of raw chunk ID hashes.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ------------------------------------------------------------------ #
#  Citations — numbered references                                     #
# ------------------------------------------------------------------ #


def render_citations(result: dict[str, Any]) -> None:
    """
    Render a references panel below the answer.

    Shows **all** opened chunks, sorted so that numbered (cited)
    entries appear first in number order (``[1]``, ``[2]``, …),
    followed by uncited chunks at the end.

    Uses ``citation_map`` (produced by ``numberize_result``) when
    available.  Falls back to raw chunk IDs for backward
    compatibility with older result dicts.
    """
    try:
        _render_citations_inner(result)
    except Exception as exc:
        st.warning(f"Citations rendering error: {type(exc).__name__}: {exc}")


def _render_citations_inner(result: dict[str, Any]) -> None:
    """Inner citations renderer (wrapped by error handler above)."""
    citation_map = result.get("citation_map", [])
    opened = result.get("opened", [])
    web_citations = result.get("web_citations", [])

    # Nothing at all — try legacy as last resort (for old result dicts
    # that have raw chunk IDs in result["citations"])
    if not citation_map and not opened and not web_citations:
        _render_citations_legacy(result)
        return

    # Build set of cited chunk_ids for quick lookup
    cited_ids = {entry["chunk_id"] for entry in citation_map}

    # Collect uncited opened chunks
    uncited_chunks: list[dict[str, Any]] = []
    for chunk in opened:
        cid = chunk.get("chunk_id", "")
        if cid and cid not in cited_ids:
            uncited_chunks.append(chunk)

    total = len(citation_map) + len(uncited_chunks) + len(web_citations)
    if total == 0:
        return

    # Build summary label
    cited_label = f"{len(citation_map)} cited" if citation_map else ""
    uncited_label = f"{len(uncited_chunks)} uncited" if uncited_chunks else ""
    web_label = f"{len(web_citations)} web" if web_citations else ""
    parts = [p for p in (cited_label, uncited_label, web_label) if p]

    with st.expander(
        f"📚 References ({', '.join(parts)})",
        expanded=True,
    ):
        # --- Numbered citations first, in number order ----------------
        for entry in citation_map:
            num = entry["number"]
            title = entry["title"]
            section = entry.get("section")
            snippet = entry.get("snippet", "")

            header = f"**[{num}]**&ensp;{title}"
            if section:
                header += f" › {section}"

            st.markdown(header)
            if snippet:
                st.caption(f'"{snippet}"')

        # --- Uncited chunks after ------------------------------------
        if uncited_chunks and citation_map:
            st.divider()

        for chunk in uncited_chunks:
            title = chunk.get("title", "Untitled")
            section = chunk.get("section")
            text = chunk.get("text", "")

            # Build snippet same way as numberize_citations
            snippet = text[:150].strip()
            if len(text) > 150:
                last_space = snippet.rfind(" ")
                if last_space > 100:
                    snippet = snippet[:last_space]
                snippet += "…"

            header = f"{title} *(uncited)*"
            if section:
                header += f" › {section}"

            st.markdown(header)
            if snippet:
                st.caption(f'"{snippet}"')

        # --- Web citations -------------------------------------------
        if web_citations:
            if citation_map or uncited_chunks:
                st.divider()
            st.markdown("**Web sources:**")
            for url in web_citations:
                st.markdown(f"- [{url}]({url})")


def _render_citations_legacy(result: dict[str, Any]) -> None:
    """
    Legacy citation rendering — raw chunk IDs.

    Used when ``citation_map`` is not present (e.g. results from
    older runs or non-numberized graphs).
    """
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


# ------------------------------------------------------------------ #
#  Timeline                                                            #
# ------------------------------------------------------------------ #


def render_timeline(result: dict[str, Any]) -> None:
    """Render timeline items from the researcher graph."""
    timeline = result.get("timeline", [])
    if not timeline:
        return

    # Build a chunk_id → number lookup from citation_map (if present)
    id_to_num = _build_id_to_number(result)

    with st.expander(f"📅 Timeline ({len(timeline)} events)", expanded=False):
        for item in timeline:
            date = item.get("date", "?")
            event = item.get("event", "")
            chunks = item.get("supporting_chunk_ids", [])

            # Convert chunk IDs to numbered references if possible
            if id_to_num and chunks:
                cite_parts = []
                for c in chunks:
                    if c in id_to_num:
                        cite_parts.append(f"[{id_to_num[c]}]")
                    else:
                        cite_parts.append(f"`{c[:12]}…`")
                cite_str = " ".join(cite_parts)
            elif chunks:
                cite_str = ", ".join(f"`{c}`" for c in chunks)
            else:
                cite_str = ""

            st.markdown(f"**{date}** — {event}")
            if cite_str:
                st.caption(f"Sources: {cite_str}")


# ------------------------------------------------------------------ #
#  Routing info (supervisor graph)                                     #
# ------------------------------------------------------------------ #


def render_routing_info(result: dict[str, Any]) -> None:
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


# ------------------------------------------------------------------ #
#  Run metadata (tokens, latency, Phoenix link)                        #
# ------------------------------------------------------------------ #


def render_run_metadata(result: dict[str, Any], run_id: str, cfg) -> None:
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


# ------------------------------------------------------------------ #
#  Evidence (opened chunks)                                            #
# ------------------------------------------------------------------ #


def render_evidence(result: dict[str, Any]) -> None:
    """
    Render opened evidence chunks.

    When ``citation_map`` is available, each chunk is labelled with
    its citation number (e.g. ``[1]``) for easy cross-referencing
    with the answer text.
    """
    opened = result.get("opened", [])
    if not opened:
        return

    id_to_num = _build_id_to_number(result)

    with st.expander(f"📄 Evidence ({len(opened)} chunks)", expanded=False):
        for chunk in opened:
            title = chunk.get("title", "Untitled")
            section = chunk.get("section", "")
            chunk_id = chunk.get("chunk_id", "?")
            text = chunk.get("text", "")

            # Build header with numbered label if available
            if chunk_id in id_to_num:
                num = id_to_num[chunk_id]
                header = f"**[{num}]** {title}"
            else:
                header = f"**{title}**  _(uncited)_"

            if section:
                header += f" › {section}"

            st.markdown(header)
            st.text(text[:500] + ("…" if len(text) > 500 else ""))
            st.divider()


# ------------------------------------------------------------------ #
#  Web results                                                         #
# ------------------------------------------------------------------ #


def render_web_results(result: dict[str, Any]) -> None:
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


# ------------------------------------------------------------------ #
#  Calculator result (supervisor math route)                           #
# ------------------------------------------------------------------ #


def render_calculator_result(result: dict[str, Any]) -> None:
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
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #


def _build_id_to_number(result: dict[str, Any]) -> dict[str, int]:
    """
    Build a chunk_id → citation number lookup from ``citation_map``.

    Returns an empty dict if citation_map is not present.
    """
    citation_map = result.get("citation_map", [])
    if not citation_map:
        return {}
    return {entry["chunk_id"]: entry["number"] for entry in citation_map}
