"""
Prompt builder: load templates, fill placeholders, track versions.

Templates live in ``src/workbench/prompting/templates/*.txt``.  Each
template uses ``{placeholder}`` syntax (Python str.format).

The builder hashes the raw template text to produce a short version tag
so every logged prompt can be traced back to the exact template that
produced it.

Usage:
    pb = PromptBuilder()
    prompt = pb.render("timeline_extract", question="...", evidence="...")
    print(pb.get_version("timeline_extract"))   # e.g. "v-3a8c1f"
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
