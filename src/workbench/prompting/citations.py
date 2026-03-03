"""
Citation extraction, validation, and numberization.

This is the single source of truth for citation parsing and policy
enforcement.  Both ``rag_graph`` and ``researcher_graph`` import
from here — no duplicated extraction logic.

Two policy presets are provided:

``RAG_DEFAULT``
    Light-touch: extract citations, no hard failures.

``RESEARCHER_STRICT``
    Requires ≥ 1 valid citation, rejects hallucinated IDs,
    enforces minimum citation density, and validates that every
    timeline item has supporting chunk IDs.  Triggers a single
    repair attempt on failure.

Numberization (post-processing):

``numberize_citations``
    Replaces raw ``[chunk_id_hash]`` references in an answer with
    sequential ``[1]``, ``[2]``, etc., ordered by first appearance.
    Returns the cleaned text plus a structured citation map for
    display in the UI.

``numberize_result``
    Convenience wrapper that applies ``numberize_citations`` to a
    graph result dict in-place.

Usage::

    from workbench.prompting.citations import (
        extract_citations,
        extract_web_citations,
        validate_citations,
        numberize_citations,
        numberize_result,
        RESEARCHER_STRICT,
    )

    cited = extract_citations(answer_text, known_ids)
    result = validate_citations(
        answer_text=answer_text,
        opened_ids=known_ids,
        policy=RESEARCHER_STRICT,
        timeline=timeline_items,
    )
    if not result.valid:
        # re-prompt / repair
        ...

    # Post-process: replace hashes with [1], [2], etc.
    clean_text, citation_map = numberize_citations(answer_text, opened_chunks)

    # Or, on a full graph result dict:
    numberize_result(graph_result)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ------------------------------------------------------------------ #
#  Policy configuration                                               #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class CitationPolicy:
    """
    Configurable rules for citation validation.

    Attributes:
        min_citations:  Minimum number of valid citations required.
        max_hallucinated:  Maximum hallucinated (unknown) IDs allowed
                           before the answer is rejected.  0 = reject any.
        min_density:  Minimum ratio of paragraphs that contain at least
                      one citation.  0.0 = no density check.
                      1.0 = every paragraph must cite something.
        require_timeline_chunk_ids:  If True, every timeline item must
                                     have non-empty ``supporting_chunk_ids``
                                     that are a subset of ``opened_ids``.
        max_repair_attempts:  How many re-prompt attempts the graph is
                              allowed before accepting the answer as-is.
        warn_on_unused_chunks:  If True, log a warning when opened chunks
                                are not cited (but don't fail).
    """

    min_citations: int = 0
    max_hallucinated: int = 0
    min_density: float = 0.0
    require_timeline_chunk_ids: bool = False
    max_repair_attempts: int = 0
    warn_on_unused_chunks: bool = False


# --- Presets -------------------------------------------------------

RAG_DEFAULT = CitationPolicy(
    min_citations=0,
    max_hallucinated=0,
    min_density=0.0,
    require_timeline_chunk_ids=False,
    max_repair_attempts=0,
    warn_on_unused_chunks=False,
)

RESEARCHER_STRICT = CitationPolicy(
    min_citations=1,
    max_hallucinated=0,
    min_density=0.5,  # at least 1 citation per 2 paragraphs
    require_timeline_chunk_ids=True,
    max_repair_attempts=1,
    warn_on_unused_chunks=True,
)


# ------------------------------------------------------------------ #
#  Validation result                                                  #
# ------------------------------------------------------------------ #


@dataclass
class ValidationResult:
    """
    Outcome of ``validate_citations()``.

    ``valid`` is True only when all policy *errors* pass.  Warnings
    are informational — they never cause ``valid`` to be False.
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Breakdown for metrics / debugging
    cited_ids: list[str] = field(default_factory=list)
    hallucinated_ids: list[str] = field(default_factory=list)
    uncited_opened_ids: list[str] = field(default_factory=list)
    paragraph_count: int = 0
    paragraphs_with_citations: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for state / logging."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "cited_ids": self.cited_ids,
            "hallucinated_ids": self.hallucinated_ids,
            "uncited_opened_ids": self.uncited_opened_ids,
            "paragraph_count": self.paragraph_count,
            "paragraphs_with_citations": self.paragraphs_with_citations,
        }


# ------------------------------------------------------------------ #
#  Extraction                                                         #
# ------------------------------------------------------------------ #


def extract_citations(answer_text: str, known_ids: list[str]) -> list[str]:
    """
    Find chunk_ids mentioned in *answer_text*.

    Looks for each *known_id* as a substring of the text (the
    convention is ``[chunk_id]`` notation, but we match the bare id
    so partial formatting still works).

    Only IDs present in *known_ids* are returned — hallucinated
    IDs are silently skipped here (use ``validate_citations`` to
    detect them).

    Args:
        answer_text: The generated answer string.
        known_ids:   List of chunk IDs that were provided as evidence.

    Returns:
        De-duplicated list of cited chunk_ids (subset of *known_ids*
        that appear in the text), preserving the order of *known_ids*.
    """
    cited: list[str] = []
    for cid in known_ids:
        if cid and cid in answer_text:
            cited.append(cid)
    return cited


def extract_web_citations(answer_text: str, urls: list[str]) -> list[str]:
    """
    Find URLs mentioned in the answer text.

    Args:
        answer_text: The generated answer string.
        urls: List of URLs from web search results.

    Returns:
        List of cited URLs found in the text.
    """
    cited: list[str] = []
    for url in urls:
        if url and url in answer_text:
            cited.append(url)
    return cited


def find_all_bracket_ids(text: str) -> list[str]:
    """
    Extract every ``[some_id]`` token from *text*.

    This captures both valid and hallucinated IDs — useful for
    detecting when the model invents chunk IDs.

    Returns:
        List of unique IDs found (order of first appearance).
    """
    seen: set[str] = set()
    result: list[str] = []
    for match in re.finditer(r"\[([A-Za-z0-9_]+)\]", text):
        cid = match.group(1)
        if cid not in seen:
            seen.add(cid)
            result.append(cid)
    return result


# ------------------------------------------------------------------ #
#  Numberization — post-process [chunk_id] → [1], [2], etc.           #
# ------------------------------------------------------------------ #

# Matches tokens like [abc123def456] — anything in brackets that
# looks like a chunk ID (hex hash or alphanumeric with underscores).
_BRACKET_TOKEN_RE = re.compile(r"\[([A-Za-z0-9_]+)\]")


def numberize_citations(
    answer_text: str,
    opened_chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Replace ``[chunk_id]`` references with sequential ``[1]``, ``[2]``, etc.

    The first chunk_id to appear in the text gets ``[1]``, the second
    unique chunk_id gets ``[2]``, and so on.  Subsequent occurrences of
    the same chunk_id reuse the same number.

    Only chunk_ids that match an entry in *opened_chunks* are
    numberized.  Unknown IDs (hallucinated or web-style) are left
    untouched.

    Example::

        # Raw LLM output:
        "Rome fell in 476 AD [b165567c4c9e5be6]. The last emperor
         was Romulus [b165567c4c9e5be6][a3f29bc1de840712]."

        # After numberize_citations():
        "Rome fell in 476 AD [1]. The last emperor
         was Romulus [1][2]."

        # citation_map:
        [
            {"number": 1, "chunk_id": "b165567c4c9e5be6",
             "title": "Fall of the Western Roman Empire", ...},
            {"number": 2, "chunk_id": "a3f29bc1de840712",
             "title": "Romulus Augustulus", ...},
        ]

    Args:
        answer_text:    The raw answer text containing ``[chunk_id]``
                        references.
        opened_chunks:  List of opened chunk dicts, each with at least
                        ``chunk_id``, ``title``, and ``text`` keys.

    Returns:
        A tuple of:
        - **numbered_text**: The answer with ``[1]``, ``[2]``, etc.
        - **citation_map**: Ordered list of dicts, one per unique cited
          chunk, with keys:

          - ``number`` (int): The citation number (1-based).
          - ``chunk_id`` (str): Original chunk ID hash.
          - ``title`` (str): Wikipedia article title.
          - ``section`` (str | None): Section heading within the article.
          - ``snippet`` (str): First ~150 chars of the chunk text.
    """
    if not answer_text or not opened_chunks:
        return answer_text, []

    # Build a lookup: chunk_id → chunk dict
    chunk_lookup: dict[str, dict[str, Any]] = {}
    for chunk in opened_chunks:
        cid = chunk.get("chunk_id", "")
        if cid:
            chunk_lookup[cid] = chunk

    # Pass 1: scan for all [token] matches in order of appearance,
    # assign numbers to known chunk_ids.  First appearance = lowest number.
    id_to_number: dict[str, int] = {}
    next_number = 1

    for match in _BRACKET_TOKEN_RE.finditer(answer_text):
        token = match.group(1)
        if token in chunk_lookup and token not in id_to_number:
            id_to_number[token] = next_number
            next_number += 1

    # Nothing to numberize
    if not id_to_number:
        return answer_text, []

    # Pass 2: replace all [chunk_id] with [N] for known IDs.
    # Process from right to left so character offsets stay valid.
    matches = list(_BRACKET_TOKEN_RE.finditer(answer_text))
    result_chars = list(answer_text)

    for match in reversed(matches):
        token = match.group(1)
        if token in id_to_number:
            num = id_to_number[token]
            replacement = f"[{num}]"
            result_chars[match.start() : match.end()] = list(replacement)

    numbered_text = "".join(result_chars)

    # Pass 3: build the citation map in number order.
    citation_map: list[dict[str, Any]] = []
    for cid, num in sorted(id_to_number.items(), key=lambda x: x[1]):
        chunk = chunk_lookup[cid]
        text = chunk.get("text", "")
        snippet = _make_snippet(text, max_len=150)

        citation_map.append(
            {
                "number": num,
                "chunk_id": cid,
                "title": chunk.get("title", "Untitled"),
                "section": chunk.get("section"),
                "snippet": snippet,
            }
        )

    return numbered_text, citation_map


def _make_snippet(text: str, max_len: int = 150) -> str:
    """Truncate text to a readable snippet, breaking at word boundaries."""
    if not text:
        return ""
    snippet = text[:max_len].strip()
    if len(text) > max_len:
        last_space = snippet.rfind(" ")
        if last_space > max_len // 2:
            snippet = snippet[:last_space]
        snippet += "…"
    return snippet


def numberize_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    Post-process a graph result dict: numberize citations in-place.

    This is the convenience wrapper meant to be called after any
    graph returns.  It reads ``answer_text`` and ``opened`` from the
    result, applies :func:`numberize_citations`, and writes back:

    - ``answer_text`` — rewritten with ``[1]``, ``[2]``, etc.
    - ``citation_map`` — ordered list of citation metadata dicts.

    The original ``citations`` list (raw chunk IDs) is preserved
    untouched for backward compatibility and logging.

    Args:
        result: The graph result dict (mutated in-place and returned).

    Returns:
        The same dict with ``answer_text`` rewritten and
        ``citation_map`` added.
    """
    answer_text = result.get("answer_text", "")
    opened = result.get("opened", [])

    if answer_text and opened:
        numbered_text, citation_map = numberize_citations(answer_text, opened)
        result["answer_text"] = numbered_text
        result["citation_map"] = citation_map
    else:
        result["citation_map"] = []

    return result


# ------------------------------------------------------------------ #
#  Validation                                                         #
# ------------------------------------------------------------------ #


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines; drop empty segments."""
    paras = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paras if p.strip()]


def validate_citations(
    answer_text: str,
    opened_ids: list[str],
    policy: CitationPolicy,
    timeline: list[dict[str, Any]] | None = None,
) -> ValidationResult:
    """
    Validate an answer's citations against a ``CitationPolicy``.

    Args:
        answer_text:  The generated answer.
        opened_ids:   Chunk IDs the agent actually opened (the
                      "allowed set").
        policy:       ``CitationPolicy`` controlling strictness.
        timeline:     Optional timeline items (list of dicts with
                      ``supporting_chunk_ids``).

    Returns:
        A ``ValidationResult`` describing pass/fail plus details.
    """
    result = ValidationResult()
    opened_set = set(opened_ids)

    # --- 1. Extract valid + hallucinated IDs ----------------------
    all_bracket = find_all_bracket_ids(answer_text)
    result.cited_ids = [cid for cid in all_bracket if cid in opened_set]
    result.hallucinated_ids = [cid for cid in all_bracket if cid not in opened_set]
    result.uncited_opened_ids = [
        cid for cid in opened_ids if cid not in set(result.cited_ids)
    ]

    # --- 2. Check minimum citations --------------------------------
    if len(result.cited_ids) < policy.min_citations:
        result.errors.append(
            f"Too few valid citations: {len(result.cited_ids)} "
            f"(need >= {policy.min_citations})"
        )

    # --- 3. Check hallucinated IDs ---------------------------------
    if len(result.hallucinated_ids) > policy.max_hallucinated:
        result.errors.append(f"Hallucinated chunk IDs: {result.hallucinated_ids}")

    # --- 4. Check citation density ---------------------------------
    if policy.min_density > 0:
        paragraphs = _split_paragraphs(answer_text)
        result.paragraph_count = len(paragraphs)
        cited_paras = 0
        for para in paragraphs:
            if any(cid in para for cid in opened_set):
                cited_paras += 1
        result.paragraphs_with_citations = cited_paras

        if paragraphs:
            density = cited_paras / len(paragraphs)
            if density < policy.min_density:
                result.errors.append(
                    f"Citation density too low: {density:.2f} "
                    f"({cited_paras}/{len(paragraphs)} paragraphs) "
                    f"(need >= {policy.min_density:.2f})"
                )

    # --- 5. Timeline citation completeness -------------------------
    if policy.require_timeline_chunk_ids and timeline:
        for i, item in enumerate(timeline):
            chunk_ids = item.get("supporting_chunk_ids", [])
            if not chunk_ids:
                result.errors.append(f"Timeline item {i} has no supporting_chunk_ids")
            else:
                invalid = [cid for cid in chunk_ids if cid not in opened_set]
                if invalid:
                    result.errors.append(
                        f"Timeline item {i} cites unknown chunks: {invalid}"
                    )

    # --- 6. Warn on unused chunks (never fails) --------------------
    if policy.warn_on_unused_chunks and result.uncited_opened_ids:
        result.warnings.append(
            f"Opened chunks not cited in answer: {result.uncited_opened_ids}"
        )

    # --- Final verdict ---------------------------------------------
    result.valid = len(result.errors) == 0
    return result
