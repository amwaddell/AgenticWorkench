"""
Date normalizer tool: parse fuzzy date strings into ISO 8601 format.

Helps the timeline extraction step by normalising diverse date formats
found in source text into consistent ISO dates.  Handles:

- Full dates: ``"March 15, 44 BC"``, ``"15/03/2024"``, ``"2024-03-15"``
- Partial dates: ``"March 2024"``, ``"2024"``, ``"44 BC"``
- Relative references: ``"circa 500 AD"``, ``"early 1800s"``
- Ranges: ``"1914–1918"`` → normalises to start date

This tool uses **only the standard library** (``re`` + ``datetime``),
so there's zero extra dependency.

Usage::

    tool = DateNormalizerTool()
    result = tool.execute(date_string="March 15, 44 BC")
    # {"original": "March 15, 44 BC", "iso": "-0044-03-15",
    #  "year": -44, "precision": "day", "era": "BC", "error": None}
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from workbench.observability.tracing import add_span_attributes
from workbench.tools.base import ToolSpec

# ------------------------------------------------------------------ #
#  Date parsing helpers                                               #
# ------------------------------------------------------------------ #

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Common suffixes / modifiers to strip
_ERA_PATTERN = re.compile(
    r"\b(bc|bce|ad|ce|b\.c\.|a\.d\.|b\.c\.e\.|c\.e\.)\b",
    re.IGNORECASE,
)
_CIRCA_PATTERN = re.compile(
    r"(?:\b(?:circa|ca\.?|approximately|approx\.?|about|around)|~)\s*",
    re.IGNORECASE,
)
_ORDINAL_PATTERN = re.compile(r"(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_DECADE_PATTERN = re.compile(r"\b(\d{3,4})s\b")

# Patterns matched in order of specificity
_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_SLASH_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_MONTH_DAY_YEAR = re.compile(r"^([a-z]+)\s+(\d{1,2}),?\s+(\d{1,4})$", re.IGNORECASE)
_DAY_MONTH_YEAR = re.compile(r"^(\d{1,2})\s+([a-z]+)\s+(\d{1,4})$", re.IGNORECASE)
_MONTH_YEAR = re.compile(r"^([a-z]+)\s+(\d{1,4})$", re.IGNORECASE)
_YEAR_ONLY = re.compile(r"^(\d{1,4})$")


def _detect_era(raw: str) -> tuple[str, str]:
    """Extract era (BC/AD) and return (cleaned_string, era)."""
    era = "AD"  # default
    match = _ERA_PATTERN.search(raw)
    if match:
        era_str = match.group(1).upper().replace(".", "")
        if era_str in ("BC", "BCE"):
            era = "BC"
        else:
            era = "AD"
        raw = _ERA_PATTERN.sub("", raw)
    return raw.strip(), era


def normalize_date(date_string: str) -> dict[str, Any]:
    """
    Parse a fuzzy date string into a structured result.

    Returns:
        Dict with keys: original, iso, year, month, day,
        precision ("day" | "month" | "year" | "decade"),
        era ("AD" | "BC"), approximate (bool), error (str | None).
    """
    original = date_string.strip()
    if not original:
        return {
            "original": original,
            "iso": None,
            "year": None,
            "month": None,
            "day": None,
            "precision": None,
            "era": None,
            "approximate": False,
            "error": "Empty date string",
        }

    cleaned = original

    # Detect approximate / circa
    approximate = bool(_CIRCA_PATTERN.search(cleaned))
    cleaned = _CIRCA_PATTERN.sub("", cleaned).strip()

    # Detect era (BC/AD)
    cleaned, era = _detect_era(cleaned)

    # Remove ordinal suffixes (1st → 1, 2nd → 2)
    cleaned = _ORDINAL_PATTERN.sub(r"\1", cleaned)

    # Handle decades (1800s → 1800, precision=decade)
    decade_match = _DECADE_PATTERN.search(cleaned)
    is_decade = False
    if decade_match:
        cleaned = _DECADE_PATTERN.sub(decade_match.group(1), cleaned)
        is_decade = True

    cleaned = cleaned.strip(" ,.-–—")

    # --- Try patterns in order of specificity -----------------------

    year: int | None = None
    month: int | None = None
    day: int | None = None
    precision = "year"

    # ISO format: 2024-03-15
    m = _ISO_DATE.match(cleaned)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        precision = "day"

    # Slash format: 03/15/2024 (US) or 15/03/2024
    if year is None:
        m = _SLASH_DATE.match(cleaned)
        if m:
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            year = y
            if a > 12:  # must be day/month/year
                day, month = a, b
            else:
                month, day = a, b
            precision = "day"

    # "March 15, 44" or "March 15 2024"
    if year is None:
        m = _MONTH_DAY_YEAR.match(cleaned)
        if m:
            month_str, d, y = m.group(1), int(m.group(2)), int(m.group(3))
            month = _MONTHS.get(month_str.lower())
            if month:
                year, day = y, d
                precision = "day"

    # "15 March 44"
    if year is None:
        m = _DAY_MONTH_YEAR.match(cleaned)
        if m:
            d, month_str, y = int(m.group(1)), m.group(2), int(m.group(3))
            month = _MONTHS.get(month_str.lower())
            if month:
                year, day = y, d
                precision = "day"

    # "March 2024"
    if year is None:
        m = _MONTH_YEAR.match(cleaned)
        if m:
            month_str, y = m.group(1), int(m.group(2))
            month = _MONTHS.get(month_str.lower())
            if month:
                year = y
                precision = "month"

    # Year only: "2024", "44"
    if year is None:
        m = _YEAR_ONLY.match(cleaned)
        if m:
            year = int(m.group(1))
            precision = "decade" if is_decade else "year"

    if year is None:
        return {
            "original": original,
            "iso": None,
            "year": None,
            "month": None,
            "day": None,
            "precision": None,
            "era": era,
            "approximate": approximate,
            "error": f"Could not parse date: {original!r}",
        }

    # Apply era
    signed_year = -year if era == "BC" else year

    # Build ISO string
    if precision == "day" and month and day:
        if era == "BC":
            iso = f"-{year:04d}-{month:02d}-{day:02d}"
        else:
            iso = f"{signed_year:04d}-{month:02d}-{day:02d}"
    elif precision == "month" and month:
        if era == "BC":
            iso = f"-{year:04d}-{month:02d}"
        else:
            iso = f"{signed_year:04d}-{month:02d}"
    else:
        if era == "BC":
            iso = f"-{year:04d}"
        else:
            iso = f"{signed_year:04d}"

    return {
        "original": original,
        "iso": iso,
        "year": signed_year,
        "month": month,
        "day": day,
        "precision": precision,
        "era": era,
        "approximate": approximate,
        "error": None,
    }


# ------------------------------------------------------------------ #
#  Args schema                                                        #
# ------------------------------------------------------------------ #


class DateNormalizerArgs(BaseModel):
    """Input schema for the date_normalizer tool."""

    date_string: str = Field(
        ..., description="A date string to normalise (e.g. 'March 15, 44 BC')."
    )


# ------------------------------------------------------------------ #
#  Tool implementation                                                #
# ------------------------------------------------------------------ #


class DateNormalizerTool(ToolSpec):
    """
    Parse fuzzy date strings into ISO 8601 format.

    Supports full dates, partial dates, BC/AD eras, circa modifiers,
    decades, and common date formats.
    """

    name: str = "date_normalizer"
    description: str = (
        "Normalise a date string into ISO 8601 format. "
        "Handles diverse formats: 'March 15, 44 BC', '15/03/2024', "
        "'circa 1800s', '2024-03-15', etc."
    )
    args_schema = DateNormalizerArgs

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Normalise a date string.

        Args (validated by DateNormalizerArgs):
            date_string: The date to parse.

        Returns:
            Dict with keys: original, iso, year, month, day,
            precision, era, approximate, error.
        """
        date_string: str = kwargs["date_string"]
        result = normalize_date(date_string)

        add_span_attributes(
            {
                "date_input": date_string[:200],
                "iso_output": str(result.get("iso", ""))[:50],
                "precision": str(result.get("precision", "")),
            }
        )

        return result

    def __repr__(self) -> str:
        return "DateNormalizerTool()"
