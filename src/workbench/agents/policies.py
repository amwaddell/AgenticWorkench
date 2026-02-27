"""
Agent policies (backward-compatibility shim).

.. deprecated::
    This module re-exports policy classes from ``workbench.legacy.policies``.
    New code should use LangGraph graphs from ``workbench.graphs`` instead.

    This shim exists so that existing tests and scripts that import from
    ``workbench.agents.policies`` continue to work without modification.
"""

from workbench.legacy.policies import (
    Action,
    ActionType,
    NeverStopPolicy,
    ResearcherPolicy,
    ScriptedPolicy,
)

__all__ = [
    "Action",
    "ActionType",
    "NeverStopPolicy",
    "ResearcherPolicy",
    "ScriptedPolicy",
]
