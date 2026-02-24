"""
Citation extraction and validation.

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

Usage::

    from workbench.prompting.citations import (
        extract_citations,
        validate_citations,
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
