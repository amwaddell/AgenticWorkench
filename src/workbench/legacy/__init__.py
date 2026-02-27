"""
Legacy orchestration: pre-LangGraph agent loop and policies.

.. deprecated::
    This package contains the original ``AgentLoop`` + policy-based
    orchestration from Days 1–7.  It has been superseded by the
    LangGraph-based graphs in ``workbench.graphs``.

    These modules are preserved so that existing tests and any
    in-flight experiments continue to work.  **Do not build new
    features on top of this code.**

Migration guide:
    AgentLoop + ScriptedPolicy   →  workbench.graphs.rag_graph
    AgentLoop + ResearcherPolicy →  workbench.graphs.researcher_graph
    (supervisor routing)         →  workbench.graphs.supervisor_graph

Import from the submodules directly::

    from workbench.legacy.loops import AgentLoop
    from workbench.legacy.policies import ScriptedPolicy, ResearcherPolicy
"""
