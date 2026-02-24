"""
Prompt builder: load templates, fill placeholders, track versions.

Templates live in ``src/workbench/prompting/templates/*.txt``.  Each
template uses ``{placeholder}`` syntax (Python str.format).

The builder hashes the raw template text to produce a short version tag
so every logged prompt can be traced back to the exact template that
produced it.

Usage (original render API)::

    pb = PromptBuilder()
    prompt = pb.render("timeline_extract", question="...", evidence="...")
    print(pb.get_version("timeline_extract"))   # e.g. "v-3a8c1f"

Usage (new build API — returns messages + metadata)::

    pb = PromptBuilder()
    messages, meta = pb.build(
        "answer_with_citations",
        question="When did Rome fall?",
        opened_chunks=opened,          # list[OpenedChunk] or list[dict]
    )
    # meta = {"variant": "answer_with_citations", "template": "...",
    #         "version": "v-...", "evidence_count": 3, ...}
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from workbench.agents.state import OpenedChunk, TimelineItem

# Default templates directory — next to this file
_DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _hash_template(text: str) -> str:
    """Create a short version hash from template text."""
    return "v-" + hashlib.sha256(text.encode()).hexdigest()[:8]


class PromptBuilder:
    """
    Load prompt templates and render them with context.

    Args:
        templates_dir: Directory containing .txt template files.
                       Defaults to the ``templates/`` folder next to
                       this module.
    """

    def __init__(self, templates_dir: Path | None = None) -> None:
        self.templates_dir = templates_dir or _DEFAULT_TEMPLATES_DIR
        # Cache: template_name → raw text
        self._cache: dict[str, str] = {}
        # Version hashes
        self._versions: dict[str, str] = {}

    def _load(self, name: str) -> str:
        """Load and cache a template by name (without .txt extension)."""
        if name in self._cache:
            return self._cache[name]

        path = self.templates_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Template not found: {path}  "
                f"(available: {[p.stem for p in self.templates_dir.glob('*.txt')]})"
            )

        text = path.read_text(encoding="utf-8")
        self._cache[name] = text
        self._versions[name] = _hash_template(text)
        return text

    def render(self, template_name: str, **kwargs: Any) -> str:
        """
        Render a template with the given placeholder values.

        Args:
            template_name: Name of the template (e.g. "timeline_extract").
            **kwargs: Values for ``{placeholder}`` substitution.

        Returns:
            Rendered prompt string.
        """
        raw = self._load(template_name)
        try:
            return raw.format(**kwargs)
        except KeyError as exc:
            raise KeyError(
                f"Missing placeholder {exc} in template '{template_name}'.  "
                f"Provided keys: {list(kwargs.keys())}"
            ) from exc

    def get_version(self, template_name: str) -> str:
        """
        Return the version hash for a loaded template.

        The template must have been loaded (via ``render`` or ``_load``)
        before calling this.
        """
        if template_name not in self._versions:
            self._load(template_name)
        return self._versions[template_name]

    def list_templates(self) -> list[str]:
        """Return names of all available templates."""
        return sorted(p.stem for p in self.templates_dir.glob("*.txt"))

    # ----------------------------------------------------------------- #
    #  High-level build API (Day 3+)                                    #
    # ----------------------------------------------------------------- #

    def build(
        self,
        variant: str,
        *,
        question: str,
        opened_chunks: list[OpenedChunk] | list[dict[str, Any]],
        timeline: list[TimelineItem] | list[dict[str, Any]] | None = None,
        extra_instructions: str | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """
        Build prompt messages and metadata for a given template variant.

        This is the recommended entry point for graph nodes and any
        future system that needs a prompt.  It normalises inputs
        (accepts both Pydantic objects and plain dicts), renders the
        template, and returns a ``(messages, meta)`` tuple so the
        caller can pass messages straight to the model and log meta
        for observability.

        Args:
            variant:  Template name (e.g. ``"answer_with_citations"``).
            question:  The user question.
            opened_chunks:  Evidence chunks — accepts ``OpenedChunk``
                            Pydantic objects **or** plain dicts (as
                            they appear in LangGraph state).
            timeline:  Optional timeline items — accepts ``TimelineItem``
                       objects or plain dicts.  Pass ``None`` or ``[]``
                       to omit the timeline section.
            extra_instructions:  Optional text prepended to the rendered
                                 prompt (for prompt variants without a
                                 new template file).

        Returns:
            ``(messages, meta)`` where:
            - **messages** is a ``list[dict]`` ready for
              ``model.generate(messages)``.
            - **meta** is a dict with keys ``variant``, ``template``,
              ``version``, ``evidence_count``, ``timeline_count``.
        """
        # --- Normalise chunks to OpenedChunk objects ----------------
        normalised_chunks: list[OpenedChunk] = []
        for c in opened_chunks:
            if isinstance(c, dict):
                normalised_chunks.append(OpenedChunk(**c))
            else:
                normalised_chunks.append(c)

        # --- Normalise timeline to TimelineItem objects --------------
        normalised_timeline: list[TimelineItem] = []
        if timeline:
            for t in timeline:
                if isinstance(t, dict):
                    normalised_timeline.append(TimelineItem(**t))
                else:
                    normalised_timeline.append(t)

        # --- Render -------------------------------------------------
        evidence = format_evidence_block(normalised_chunks)
        timeline_section = format_timeline_section(normalised_timeline)

        user_prompt = self.render(
            variant,
            question=question,
            evidence=evidence,
            timeline_section=timeline_section,
        )

        if extra_instructions:
            user_prompt = f"{extra_instructions}\n\n{user_prompt}"

        messages = [{"role": "user", "content": user_prompt}]

        meta: dict[str, Any] = {
            "variant": variant,
            "template": f"{variant}.txt",
            "version": self.get_version(variant),
            "evidence_count": len(normalised_chunks),
            "timeline_count": len(normalised_timeline),
        }

        return messages, meta


# --------------------------------------------------------------------- #
#  Convenience helpers for common prompt patterns                        #
# --------------------------------------------------------------------- #


def format_evidence_block(opened: list[OpenedChunk]) -> str:
    """
    Format a list of opened chunks into a numbered evidence block.

    Each chunk gets a header line with its chunk_id, title, and section,
    followed by the full text.

    Args:
        opened: List of OpenedChunk objects.

    Returns:
        Formatted evidence string.
    """
    parts: list[str] = []
    for i, chunk in enumerate(opened, 1):
        header = f"[{chunk.chunk_id}] {chunk.title}"
        if chunk.section:
            header += f" > {chunk.section}"
        parts.append(f"--- Evidence {i}: {header} ---\n{chunk.text}")
    return "\n\n".join(parts)


def format_timeline_section(timeline: list[TimelineItem]) -> str:
    """
    Format a timeline into a text section for inclusion in the answer prompt.

    Returns an empty string if the timeline is empty, so the answer
    prompt template can include ``{timeline_section}`` unconditionally.

    Args:
        timeline: List of TimelineItem objects.

    Returns:
        Formatted timeline section, or empty string.
    """
    if not timeline:
        return ""

    lines = ["--- TIMELINE (for reference while writing your answer) ---"]
    for item in timeline:
        chunk_refs = ", ".join(item.supporting_chunk_ids) or "no source"
        lines.append(f"• {item.date}: {item.event}  [{chunk_refs}]")
    lines.append("")  # trailing newline
    return "\n".join(lines)
