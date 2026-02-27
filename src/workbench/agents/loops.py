"""
Agent execution loop (backward-compatibility shim).

.. deprecated::
    This module re-exports ``AgentLoop`` from ``workbench.legacy.loops``.
    New code should use LangGraph graphs from ``workbench.graphs`` instead.

    This shim exists so that existing tests and scripts that import from
    ``workbench.agents.loops`` continue to work without modification.
"""

from workbench.legacy.loops import AgentLoop

__all__ = ["AgentLoop"]
