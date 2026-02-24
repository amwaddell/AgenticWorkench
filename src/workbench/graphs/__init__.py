"""
LangGraph-based workflow graphs.

This package contains compiled LangGraph graphs that replace the
sequential ``AgentLoop`` with explicit node-and-edge DAGs.

Graphs
------
rag_graph          — Day 3: search → open → answer (ports ScriptedPolicy)
researcher_graph   — Day 4: search → open → timeline → answer → validate
                     (ports ResearcherPolicy with stricter citation discipline)
"""

from workbench.graphs.base_state import RAGState, ResearcherState
from workbench.graphs.rag_graph import (
    build_rag_graph,
    extract_citations,
    run_rag_graph,
)
from workbench.graphs.researcher_graph import (
    build_researcher_graph,
    run_researcher_graph,
)

__all__ = [
    "RAGState",
    "ResearcherState",
    "build_rag_graph",
    "build_researcher_graph",
    "extract_citations",
    "run_rag_graph",
    "run_researcher_graph",
]
