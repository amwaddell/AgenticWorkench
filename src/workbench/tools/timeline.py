"""
Timeline tool: extract a chronological timeline from evidence chunks.

The tool calls the language model with a structured-output prompt and
parses the JSON response into a list of ``TimelineItem`` objects.

Migrated to ``ToolSpec`` (Day 2): args are validated via Pydantic,
and tracing / logging / metrics hooks are automatic.

Usage (direct):
    tool = TimelineTool(model=llm, prompt_builder=pb)
    result = tool.execute(
        question="When did the Roman Republic end?",
        opened_chunks=[...],
    )

Usage (LangGraph):
    lc_tool = tool.to_langchain_tool()
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from workbench.agents.state import OpenedChunk, TimelineItem
from workbench.observability.tracing import add_span_attributes
from workbench.prompting.prompt_builder import PromptBuilder, format_evidence_block
from workbench.tools.base import ToolSpec

# ------------------------------------------------------------------ #
#  Args schema                                                        #
# ------------------------------------------------------------------ #


class TimelineArgs(BaseModel):
    """Input schema for the timeline tool.

    ``opened_chunks`` accepts both plain dicts (from LangGraph JSON)
    and ``OpenedChunk`` Pydantic objects (from the agent loop).
    """

    question: str = Field(..., description="The research question.")
    opened_chunks: list[Any] = Field(
        ...,
        description=(
            "Evidence chunks — list of dicts or OpenedChunk objects "
            "with keys: chunk_id, document_id, title, section, text."
        ),
    )


# ------------------------------------------------------------------ #
#  Tool implementation                                                #
# ------------------------------------------------------------------ #


class TimelineTool(ToolSpec):
    """
    Extract a date-ordered timeline of events from opened evidence.

    Args:
        model: LanguageModel with ``.generate(messages, **settings)``.
        prompt_builder: PromptBuilder instance (loads timeline_extract.txt).
        generation_settings: Extra kwargs forwarded to ``model.generate``.
    """

    name: str = "timeline"
    description: str = (
        "Extract a chronological timeline of events from evidence chunks. "
        "Each event is tied to one or more supporting chunk_ids."
    )
    args_schema = TimelineArgs

    def __init__(
        self,
        model: Any,
        prompt_builder: PromptBuilder | None = None,
        generation_settings: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.generation_settings = generation_settings or {}

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Extract a timeline from evidence chunks.

        Args (validated by TimelineArgs):
            question: The research question.
            opened_chunks: Evidence chunks as list of dicts.

        Returns:
            Dict with keys: timeline, count, cited_chunk_ids,
            raw_model_output, prompt_version.
        """
        question: str = kwargs["question"]
        raw_chunks: list[dict[str, Any]] = kwargs["opened_chunks"]

        # Convert dicts to OpenedChunk objects
        opened: list[OpenedChunk] = []
        for c in raw_chunks:
            if isinstance(c, OpenedChunk):
                opened.append(c)
            elif isinstance(c, dict):
                opened.append(OpenedChunk(**c))

        # Build the prompt
        evidence = format_evidence_block(opened)
        user_prompt = self.prompt_builder.render(
            "timeline_extract",
            question=question,
            evidence=evidence,
        )
        prompt_version = self.prompt_builder.get_version("timeline_extract")

        messages = [{"role": "user", "content": user_prompt}]

        # Call model
        response = self.model.generate(messages, **self.generation_settings)

        # Parse response
        timeline_items = self._parse_timeline(
            response.text,
            known_chunk_ids={c.chunk_id for c in opened},
        )

        # Collect all cited chunk_ids
        cited: set[str] = set()
        for item in timeline_items:
            cited.update(item.supporting_chunk_ids)

        add_span_attributes(
            {
                "timeline_count": len(timeline_items),
                "cited_chunk_count": len(cited),
                "prompt_version": prompt_version,
            }
        )

        return {
            "timeline": [item.model_dump() for item in timeline_items],
            "count": len(timeline_items),
            "cited_chunk_ids": sorted(cited),
            "raw_model_output": response.text,
            "prompt_version": prompt_version,
        }

    # ------------------------------------------------------------------ #
    #  Parsing helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_timeline(
        raw_text: str,
        known_chunk_ids: set[str] | None = None,
    ) -> list[TimelineItem]:
        """
        Parse model output into TimelineItem objects.

        Handles common model quirks:
        - Markdown code fences around JSON
        - Trailing commas
        - Extra whitespace

        Args:
            raw_text: Raw model output.
            known_chunk_ids: If provided, filter supporting_chunk_ids
                             to only those in this set.

        Returns:
            List of validated TimelineItem objects.

        Raises:
            ValueError: If the output cannot be parsed at all.
        """
        cleaned = raw_text.strip()

        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # Handle empty timeline
        if cleaned in ("[]", ""):
            return []

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Try removing trailing commas (common LLM mistake)
            fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Could not parse timeline JSON: {exc}\n"
                    f"Raw output (first 500 chars): {raw_text[:500]}"
                ) from exc

        if not isinstance(data, list):
            raise ValueError(
                f"Expected JSON array, got {type(data).__name__}: {str(data)[:200]}"
            )

        items: list[TimelineItem] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue  # skip malformed entries

            # Require at minimum date and event
            if "date" not in entry or "event" not in entry:
                continue

            chunk_ids = entry.get("supporting_chunk_ids", [])
            if not isinstance(chunk_ids, list):
                chunk_ids = [str(chunk_ids)]

            # Filter to known chunk_ids if provided
            if known_chunk_ids is not None:
                chunk_ids = [cid for cid in chunk_ids if cid in known_chunk_ids]

            items.append(
                TimelineItem(
                    date=str(entry["date"]),
                    event=str(entry["event"]),
                    supporting_chunk_ids=chunk_ids,
                )
            )

        return items

    def __repr__(self) -> str:
        return f"TimelineTool(model={self.model!r})"
