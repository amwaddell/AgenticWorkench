"""
LangGraph-based workflow graphs.

This package contains compiled LangGraph graphs that replace the
sequential ``AgentLoop`` with explicit node-and-edge DAGs.

Graphs
------
rag_graph          — Day 3: search → open → answer (ports ScriptedPolicy)
                     Day 5: optional web search fallback branch
researcher_graph   — Day 4: search → open → timeline → answer → validate
                     (ports ResearcherPolicy with stricter citation discipline)
supervisor_graph   — Day 7: hierarchical routing to specialist subgraphs
                     (local RAG, web research, timeline, math)

Subpackages
-----------
routing            — Day 7: query classification (heuristic + LLM fallback)
"""

from workbench.graphs.base_state import RAGState, ResearcherState, SupervisorState
from workbench.graphs.rag_graph import (
    build_rag_graph,
    extract_citations,
    extract_web_citations,
    run_rag_graph,
)
from workbench.graphs.researcher_graph import (
    build_researcher_graph,
    run_researcher_graph,
)
from workbench.graphs.routing import heuristic_route, llm_route
from workbench.graphs.supervisor_graph import (
    build_supervisor_graph,
    run_supervisor_graph,
)

__all__ = [
    "RAGState",
    "ResearcherState",
    "SupervisorState",
    "build_rag_graph",
    "build_researcher_graph",
    "build_supervisor_graph",
    "extract_citations",
    "extract_web_citations",
    "heuristic_route",
    "llm_route",
    "run_rag_graph",
    "run_researcher_graph",
    "run_supervisor_graph",
]
