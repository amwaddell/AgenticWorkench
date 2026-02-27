"""
Systems: high-level entry points for running the workbench.

GraphRunner / run_graph
    The recommended way to run any graph with full observability.

Usage::

    from workbench.systems.runner import GraphRunner

    runner = GraphRunner()
    result = runner.run("rag_graph", "When did the Roman Republic end?")
    print(result["answer_text"])
"""

from workbench.systems.runner import GraphRunner, run_graph

__all__ = [
    "GraphRunner",
    "run_graph",
]
