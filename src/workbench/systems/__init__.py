"""
Systems: high-level entry points for running the workbench.

GraphRunner / run_graph
    The recommended way to run any graph with full observability.
    Replaces the legacy ``ChatbotSystem`` for new code.

ChatbotSystem (legacy)
    Still works but uses the old ``AgentLoop`` internally.
    Prefer ``GraphRunner`` for new projects.
"""

from workbench.systems.runner import GraphRunner, run_graph

__all__ = [
    "GraphRunner",
    "run_graph",
]
