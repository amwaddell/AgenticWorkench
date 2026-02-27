"""
Heuristic (rule-based) query router.

Fast, deterministic, and fully unit-testable.  Returns a confident
route for obvious cases or ``ROUTE_AMBIGUOUS`` when the rules can't
decide — letting the LLM fallback handle it in hybrid mode.

Add new rules here as you discover patterns that the heuristic
misclassifies.  Each rule should be one line of intent so the
file stays scannable.
"""

from __future__ import annotations

import re

from workbench.graphs.routing.constants import (
    ROUTE_AMBIGUOUS,
    ROUTE_LOCAL_RAG,
    ROUTE_MATH,
    ROUTE_TIMELINE,
    ROUTE_WEB_RESEARCH,
)

# ------------------------------------------------------------------ #
#  Pattern tables                                                     #
# ------------------------------------------------------------------ #

# Pure arithmetic expression (digits + operators only)
_MATH_EXPR_RE = re.compile(r"^\s*[\d\(\)\+\-\*\/\.\s\^%]+\s*$")

# Strong math verbs — route to math even without digits
_MATH_STRONG_KEYWORDS: set[str] = {
    "calculate",
    "compute",
    "solve",
    "evaluate",
    "square root",
    "factorial",
    "logarithm",
}

# Weak math phrases — only route to math when digits are present,
# otherwise they're too greedy ("what is photosynthesis?" ≠ math)
_MATH_WEAK_KEYWORDS: set[str] = {
    "what is",
    "how much is",
    "how many is",
    "sum of",
    "product of",
    "difference between",
    "percent of",
    "percentage",
}

_WEB_KEYWORDS: set[str] = {
    "latest",
    "today",
    "current",
    "recent",
    "news",
    "price",
    "stock",
    "weather",
    "release date",
    "2025",
    "2026",
    "2027",
    "right now",
    "this week",
    "this month",
    "this year",
    "who is the current",
    "what is the latest",
    "breaking",
    "update",
    "announced",
}

_TIMELINE_KEYWORDS: set[str] = {
    "timeline",
    "chronology",
    "chronological",
    "what happened when",
    "sequence of events",
    "dates of",
    "history of",
    "when did",
    "evolution of",
    "rise and fall",
    "from when to when",
    "over the years",
    "century",
    "decade",
    "era",
}

# Question-word opener → at least it's a factual query
_QUESTION_WORD_RE = re.compile(
    r"^(who|what|where|why|how|which|is|are|was|were|did|do|does)\b"
)


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #


def heuristic_route(question: str) -> tuple[str, float]:
    """
    Classify a question using pattern-matching heuristics.

    Check order matters — more-specific categories are tested first:

    1. Pure math expressions (digits + operators only)
    2. Strong math verbs (calculate, solve …) — always math
    3. Web research keywords (latest, today, price …)
    4. Timeline keywords (timeline, chronology, history of …)
    5. Weak math phrases (what is, how much …) — only if digits present
    6. Question-word opener → local RAG
    7. Ambiguous

    Returns:
        ``(route_label, confidence)`` where confidence is 0.0–1.0.
        Returns ``(ROUTE_AMBIGUOUS, 0.0)`` when the rules can't decide.
    """
    q_lower = question.lower().strip()
    has_digits = bool(re.search(r"\d", question))

    # --- 1. Math: pure numeric expression -----------------------------
    if _MATH_EXPR_RE.match(q_lower):
        return ROUTE_MATH, 1.0

    # --- 2. Math: strong verb keywords (always math) ------------------
    for kw in _MATH_STRONG_KEYWORDS:
        if kw in q_lower:
            return ROUTE_MATH, 0.9 if has_digits else 0.8

    # --- 3. Web research: recency / live-data signals -----------------
    for kw in _WEB_KEYWORDS:
        if kw in q_lower:
            return ROUTE_WEB_RESEARCH, 0.8

    # --- 4. Timeline: chronological intent ----------------------------
    for kw in _TIMELINE_KEYWORDS:
        if kw in q_lower:
            return ROUTE_TIMELINE, 0.8

    # --- 5. Math: weak phrases — only with digits present -------------
    if has_digits:
        for kw in _MATH_WEAK_KEYWORDS:
            if kw in q_lower:
                return ROUTE_MATH, 0.8

    # --- 6. Default: factual question → local RAG ---------------------
    if _QUESTION_WORD_RE.match(q_lower):
        return ROUTE_LOCAL_RAG, 0.5

    # --- 7. Ambiguous: heuristic gives up -----------------------------
    return ROUTE_AMBIGUOUS, 0.0
