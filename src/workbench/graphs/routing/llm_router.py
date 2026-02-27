"""
LLM-based query router (fallback).

Only called when the heuristic router returns ``ROUTE_AMBIGUOUS``
(or when ``routing_mode="llm_only"``).  Prompts the model to emit
a single classification label from the allowed set.

Designed to be cheap: low temperature, tiny max_tokens, simple
system prompt.  The model's response is sanitised and validated
against ``VALID_ROUTES`` before being accepted.
"""

from __future__ import annotations

from typing import Any

from workbench.graphs.routing.constants import DEFAULT_ROUTE, VALID_ROUTES

# ------------------------------------------------------------------ #
#  Prompt                                                             #
# ------------------------------------------------------------------ #

_ROUTER_SYSTEM_PROMPT = """\
You are a query classifier.  Given a user question, respond with EXACTLY \
one label from the following list and nothing else:

- local_rag        — factual, encyclopaedic, or historical question answerable from a local Wikipedia knowledge base
- web_research     — needs current/recent information from the internet
- timeline_research — asks about a sequence of events, chronology, or historical timeline
- math             — requires arithmetic or mathematical computation

Respond with ONLY the label.  No explanation, no punctuation.\
"""


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #


def llm_route(
    question: str,
    model: Any,
    generation_settings: dict[str, Any] | None = None,
) -> tuple[str, float]:
    """
    Classify a question by prompting the LLM.

    Args:
        question: The user's query string.
        model: A language model with ``.generate(messages, **settings)``.
        generation_settings: Extra kwargs forwarded to ``model.generate()``.

    Returns:
        ``(route_label, confidence)``.
        Confidence is 0.7 on success (moderate trust), lower on error.
    """
    gen_settings = {**(generation_settings or {}), "temperature": 0.1, "max_tokens": 20}

    messages = [
        {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        response = model.generate(messages, **gen_settings)
        raw_label = (
            response.text.strip()
            .lower()
            .replace('"', "")
            .replace("'", "")
            .split("\n")[0]
            .split(".")[0]
            .strip()
        )

        if raw_label in VALID_ROUTES:
            return raw_label, 0.7

        # Model returned something unexpected
        return DEFAULT_ROUTE, 0.3

    except Exception:
        # Model call failed entirely
        return DEFAULT_ROUTE, 0.2
