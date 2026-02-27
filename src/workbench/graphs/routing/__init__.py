"""
Query routing for the supervisor graph.

Submodules
----------
constants    — route label strings (``ROUTE_MATH``, etc.)
heuristic    — fast, deterministic, regex + keyword router
llm_router   — LLM-based fallback classifier
"""

from workbench.graphs.routing.constants import (
    DEFAULT_ROUTE,
    ROUTE_AMBIGUOUS,
    ROUTE_LOCAL_RAG,
    ROUTE_MATH,
    ROUTE_TIMELINE,
    ROUTE_WEB_RESEARCH,
    VALID_ROUTES,
)
from workbench.graphs.routing.heuristic import heuristic_route
from workbench.graphs.routing.llm_router import llm_route

__all__ = [
    "DEFAULT_ROUTE",
    "ROUTE_AMBIGUOUS",
    "ROUTE_LOCAL_RAG",
    "ROUTE_MATH",
    "ROUTE_TIMELINE",
    "ROUTE_WEB_RESEARCH",
    "VALID_ROUTES",
    "heuristic_route",
    "llm_route",
]
