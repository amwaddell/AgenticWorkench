"""
Agent policies (legacy — moved from workbench.agents.policies).

.. deprecated::
    Superseded by LangGraph-based graphs in ``workbench.graphs``.
    Kept for backward compatibility with existing tests.

A policy inspects the current AgentState and returns the next Action
for the loop to execute.

Policies
--------
ScriptedPolicy       — 3-step: search → open → answer
ResearcherPolicy     — 4-step: search → open → build timeline → answer
NeverStopPolicy      — testing only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from workbench.agents.state import AgentState


class ActionType(str, Enum):
    """Types of actions the agent can take."""

    SEARCH = "search"
    OPEN_CHUNKS = "open_chunks"
    BUILD_TIMELINE = "build_timeline"
    GENERATE_ANSWER = "generate_answer"
    STOP = "stop"


@dataclass
class Action:
    """An action for the loop to execute."""

    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)


class ScriptedPolicy:
    """
    Fixed 3-step policy.

    Step sequence:
        1. search(query=question)
        2. open_chunks(chunk_ids=top N from retrieved)
        3. generate_answer
        4. stop

    Args:
        open_top_n: How many of the retrieved chunks to open (default 3).
        search_top_k: How many chunks to retrieve (default 5).
    """

    def __init__(
        self,
        open_top_n: int = 3,
        search_top_k: int = 5,
    ) -> None:
        self.open_top_n = open_top_n
        self.search_top_k = search_top_k

    def next_action(self, state: AgentState) -> Action:
        """Decide the next action based on current state."""
        if state.done or state.error:
            return Action(action_type=ActionType.STOP)

        if not state.retrieved:
            return Action(
                action_type=ActionType.SEARCH,
                params={
                    "query": state.question,
                    "top_k": self.search_top_k,
                },
            )

        if not state.opened:
            chunk_ids = [r.chunk_id for r in state.retrieved[: self.open_top_n]]
            return Action(
                action_type=ActionType.OPEN_CHUNKS,
                params={"chunk_ids": chunk_ids},
            )

        if not state.answer_text:
            return Action(action_type=ActionType.GENERATE_ANSWER)

        return Action(action_type=ActionType.STOP)


class ResearcherPolicy:
    """
    Historical researcher: 4-step policy with timeline extraction.

    Step sequence:
        1. search(query=question)
        2. open_chunks(chunk_ids=top N from retrieved)
        3. build_timeline from evidence
        4. generate_answer with citations
        5. stop

    Args:
        open_top_n: How many of the retrieved chunks to open (default 5).
        search_top_k: How many chunks to retrieve (default 10).
    """

    def __init__(
        self,
        open_top_n: int = 5,
        search_top_k: int = 10,
    ) -> None:
        self.open_top_n = open_top_n
        self.search_top_k = search_top_k

    def next_action(self, state: AgentState) -> Action:
        """Decide the next action based on current state."""
        if state.done or state.error:
            return Action(action_type=ActionType.STOP)

        if not state.retrieved:
            return Action(
                action_type=ActionType.SEARCH,
                params={
                    "query": state.question,
                    "top_k": self.search_top_k,
                },
            )

        if not state.opened:
            chunk_ids = [r.chunk_id for r in state.retrieved[: self.open_top_n]]
            return Action(
                action_type=ActionType.OPEN_CHUNKS,
                params={"chunk_ids": chunk_ids},
            )

        if not state.timeline:
            return Action(action_type=ActionType.BUILD_TIMELINE)

        if not state.answer_text:
            return Action(action_type=ActionType.GENERATE_ANSWER)

        return Action(action_type=ActionType.STOP)


class NeverStopPolicy:
    """
    A policy that never returns STOP.

    Used only in tests to verify that the loop's max_steps guard works.
    """

    def next_action(self, state: AgentState) -> Action:
        return Action(
            action_type=ActionType.SEARCH,
            params={"query": state.question, "top_k": 1},
        )
