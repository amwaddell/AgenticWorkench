"""
Timeline tool: extract a chronological timeline from evidence chunks.

The tool calls the language model with a structured-output prompt and
parses the JSON response into a list of ``TimelineItem`` objects.

Satisfies the ``Tool`` protocol from ``workbench.core.interfaces``.

Usage:
    tool = TimelineTool(model=llm, prompt_builder=pb)
    result = tool.execute(
        question="When did the Roman Republic end?",
        opened_chunks=[...],
    )
    print(result["timeline"])   # list of TimelineItem dicts
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from workbench.agents.state import OpenedChunk, TimelineItem
from workbench.observability.tracing import add_span_attributes, start_span
from workbench.prompting.prompt_builder import PromptBuilder, format_evidence_block


class TimelineTool:
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

    def __init__(
        self,
        model: Any,
        prompt_builder: PromptBuilder | None = None,
        generation_settings: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.generation_settings = generation_settings or {}

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Extract a timeline.

        Kwargs:
            question (str): The research question.
            opened_chunks (list[OpenedChunk | dict]): Evidence chunks.

        Returns:
            Dict with keys:
                timeline: list of TimelineItem dicts
                count: number of items
                cited_chunk_ids: deduplicated list of cited chunks
                duration_ms: latency
                raw_model_output: the raw text from the model
                prompt_version: hash of the template used
        """
        question: str = kwargs.get("question", "")
        raw_chunks = kwargs.get("opened_chunks", [])

        # Accept both OpenedChunk objects and plain dicts
        opened: list[OpenedChunk] = []
        for c in raw_chunks:
            if isinstance(c, OpenedChunk):
                opened.append(c)
            elif isinstance(c, dict):
                opened.append(OpenedChunk(**c))

        with start_span(
            "tool.timeline",
            attributes={
                "question": question[:200],
                "evidence_count": len(opened),
            },
        ):
            t0 = time.time()

            # Build the prompt
            evidence = format_evidence_block(opened)
            user_prompt = self.prompt_builder.render(
                "timeline_extract",
                question=question,
                evidence=evidence,
            )
            prompt_version = self.prompt_builder.get_version("timeline_extract")

            messages = [
                {"role": "user", "content": user_prompt},
            ]

            # Call model
            response = self.model.generate(messages, **self.generation_settings)

            # Parse response
            timeline_items = self._parse_timeline(
                response.text,
                known_chunk_ids={c.chunk_id for c in opened},
            )

            duration_ms = (time.time() - t0) * 1000

            # Collect all cited chunk_ids
            cited: set[str] = set()
            for item in timeline_items:
                cited.update(item.supporting_chunk_ids)

            add_span_attributes(
                {
                    "timeline_count": len(timeline_items),
                    "cited_chunk_count": len(cited),
                    "duration_ms": round(duration_ms, 1),
                    "prompt_version": prompt_version,
                }
            )

            return {
                "timeline": [item.model_dump() for item in timeline_items],
                "count": len(timeline_items),
                "cited_chunk_ids": sorted(cited),
                "duration_ms": round(duration_ms, 1),
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
