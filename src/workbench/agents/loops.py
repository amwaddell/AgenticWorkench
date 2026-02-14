"""
Agent execution loop.

The loop is the runtime engine that:
    1. Asks the policy for the next action
    2. Executes it (calling tools or the model)
    3. Updates agent state
    4. Repeats until STOP or max_steps

Every step produces a trace span, log event, and metrics record.

Usage:
    loop = AgentLoop(
        policy=ScriptedPolicy(),
        search_tool=search_tool,
        open_chunk_tool=open_chunk_tool,
        model=llm,
        max_steps=10,
    )
    final_state = loop.run(AgentState(question="..."))
"""

from __future__ import annotations

from typing import Any

from workbench.agents.policies import Action, ActionType
from workbench.agents.state import (
    AgentState,
    OpenedChunk,
    RetrievedChunkSummary,
)
from workbench.observability.tracing import add_span_attributes, start_span

# Default answer prompt — inline for now; will move to prompting/ later.
_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer the user's question "
    "using ONLY the evidence provided below. For each claim, cite the "
    "source by writing [chunk_id] immediately after the claim.\n\n"
    "If the evidence does not contain enough information, say so honestly."
)


def _build_evidence_block(opened: list[OpenedChunk]) -> str:
    """Format opened chunks into a numbered evidence block."""
    parts: list[str] = []
    for i, chunk in enumerate(opened, 1):
        header = f"[{chunk.chunk_id}] {chunk.title}"
        if chunk.section:
            header += f" > {chunk.section}"
        parts.append(f"--- Evidence {i}: {header} ---\n{chunk.text}")
    return "\n\n".join(parts)


class AgentLoop:
    """
    Execute an agent policy step-by-step.

    Args:
        policy: Object with ``next_action(state) -> Action``.
        search_tool: SearchWikipediaTool (or any tool with ``.execute()``).
        open_chunk_tool: OpenChunkTool (or any tool with ``.execute()``).
        model: LanguageModel with ``.generate(messages, **settings)``.
        max_steps: Hard limit on loop iterations (safety guard).
        generation_settings: Extra kwargs forwarded to ``model.generate``.
    """

    def __init__(
        self,
        policy: Any,
        search_tool: Any,
        open_chunk_tool: Any,
        model: Any,
        max_steps: int = 10,
        generation_settings: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.search_tool = search_tool
        self.open_chunk_tool = open_chunk_tool
        self.model = model
        self.max_steps = max_steps
        self.generation_settings = generation_settings or {}

    def run(self, state: AgentState) -> AgentState:
        """
        Run the agent loop to completion.

        Args:
            state: Initial agent state (must have ``question`` set).

        Returns:
            Final agent state with answer populated (or error set).
        """
        with start_span(
            "agent.run",
            attributes={
                "question": state.question[:200],
                "max_steps": self.max_steps,
            },
        ):
            while state.step < self.max_steps and not state.done:
                state.step += 1
                action = self.policy.next_action(state)

                if action.action_type == ActionType.STOP:
                    state.done = True
                    break

                self._execute_step(state, action)

            # If we hit max_steps without finishing, mark it
            if not state.done:
                state.done = True
                if not state.answer_text:
                    state.error = f"max_steps reached ({self.max_steps})"

            add_span_attributes(
                {
                    "total_steps": state.step,
                    "retrieved_count": len(state.retrieved),
                    "opened_count": len(state.opened),
                    "answer_length": len(state.answer_text),
                    "has_error": state.error is not None,
                }
            )

            return state

    def _execute_step(self, state: AgentState, action: Action) -> None:
        """Execute a single step and update state."""
        with start_span(
            "agent.step",
            attributes={
                "step_number": state.step,
                "action_type": action.action_type.value,
            },
        ):
            try:
                if action.action_type == ActionType.SEARCH:
                    self._do_search(state, action)
                elif action.action_type == ActionType.OPEN_CHUNKS:
                    self._do_open_chunks(state, action)
                elif action.action_type == ActionType.GENERATE_ANSWER:
                    self._do_generate(state)
                # STOP is handled in the run() loop above
            except Exception as e:
                state.error = f"{type(e).__name__}: {e}"
                state.done = True

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _do_search(self, state: AgentState, action: Action) -> None:
        """Execute search action."""
        with start_span(
            "tool.search_wikipedia",
            attributes={"query": action.params.get("query", "")[:200]},
        ):
            result = self.search_tool.execute(**action.params)

            summaries = []
            for chunk_dict in result.get("chunks", []):
                summaries.append(RetrievedChunkSummary(**chunk_dict))

            state.retrieved = summaries

            add_span_attributes({"results_count": len(summaries)})

    def _do_open_chunks(self, state: AgentState, action: Action) -> None:
        """Execute open-chunks action."""
        chunk_ids: list[str] = action.params.get("chunk_ids", [])

        for cid in chunk_ids:
            with start_span(
                "tool.open_chunk",
                attributes={"chunk_id": cid},
            ):
                result = self.open_chunk_tool.execute(chunk_id=cid)
                if result.get("found") and result.get("chunk"):
                    state.opened.append(OpenedChunk(**result["chunk"]))

    def _do_generate(self, state: AgentState) -> None:
        """Build prompt from evidence and call the model."""
        evidence_block = _build_evidence_block(state.opened)

        user_message = (
            f"Evidence:\n{evidence_block}\n\n"
            f"Question: {state.question}\n\n"
            "Answer the question using the evidence above. "
            "Cite sources using [chunk_id] notation."
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        response = self.model.generate(messages, **self.generation_settings)

        state.answer_text = response.text
        state.tokens_in = response.tokens_in
        state.tokens_out = response.tokens_out
        state.model_latency_ms = response.latency_ms

        # Extract cited chunk_ids from answer text (simple pattern match)
        state.citations = self._extract_citations(
            response.text, [o.chunk_id for o in state.opened]
        )

    @staticmethod
    def _extract_citations(answer_text: str, known_ids: list[str]) -> list[str]:
        """
        Find chunk_ids mentioned in the answer text.

        Looks for patterns like [chunk_id] in the text.
        Only returns IDs that are in the known set.
        """
        cited: list[str] = []
        for cid in known_ids:
            if cid in answer_text:
                cited.append(cid)
        return cited
