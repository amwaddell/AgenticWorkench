"""
Tests for the DateNormalizerTool (Day 5).

Validates:
- ISO date parsing
- Slash date parsing (US format)
- Month-day-year, day-month-year, month-year formats
- Year-only
- BC/AD era detection
- Circa / approximate detection
- Decade parsing (1800s)
- Ordinal suffixes (1st, 2nd, etc.)
- Edge cases (empty, unparseable)
- ToolSpec integration
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from workbench.tools.utility.dates import (
    DateNormalizerArgs,
    DateNormalizerTool,
    normalize_date,
)

# ------------------------------------------------------------------ #
#  normalize_date unit tests                                          #
# ------------------------------------------------------------------ #


class TestNormalizeDateISO:
    """ISO 8601 format: YYYY-MM-DD."""

    def test_standard_iso(self) -> None:
        r = normalize_date("2024-03-15")
        assert r["iso"] == "2024-03-15"
        assert r["year"] == 2024
        assert r["month"] == 3
        assert r["day"] == 15
        assert r["precision"] == "day"

    def test_iso_single_digit_month(self) -> None:
        r = normalize_date("2024-3-5")
        assert r["year"] == 2024
        assert r["month"] == 3
        assert r["day"] == 5


class TestNormalizeDateSlash:
    """Slash format: MM/DD/YYYY."""

    def test_us_slash(self) -> None:
        r = normalize_date("03/15/2024")
        assert r["year"] == 2024
        assert r["month"] == 3
        assert r["day"] == 15
        assert r["precision"] == "day"

    def test_day_first_when_ambiguous(self) -> None:
        # 25 > 12, so must be day/month/year
        r = normalize_date("25/03/2024")
        assert r["day"] == 25
        assert r["month"] == 3


class TestNormalizeDateTextual:
    """Textual date formats."""

    def test_month_day_year(self) -> None:
        r = normalize_date("March 15, 2024")
        assert r["year"] == 2024
        assert r["month"] == 3
        assert r["day"] == 15
        assert r["precision"] == "day"

    def test_month_day_year_no_comma(self) -> None:
        r = normalize_date("March 15 2024")
        assert r["year"] == 2024
        assert r["month"] == 3

    def test_day_month_year(self) -> None:
        r = normalize_date("15 March 2024")
        assert r["year"] == 2024
        assert r["month"] == 3
        assert r["day"] == 15

    def test_month_year(self) -> None:
        r = normalize_date("March 2024")
        assert r["year"] == 2024
        assert r["month"] == 3
        assert r["day"] is None
        assert r["precision"] == "month"

    def test_abbreviated_month(self) -> None:
        r = normalize_date("Sep 2024")
        assert r["month"] == 9

    def test_year_only(self) -> None:
        r = normalize_date("2024")
        assert r["year"] == 2024
        assert r["month"] is None
        assert r["day"] is None
        assert r["precision"] == "year"


class TestNormalizeDateEra:
    """BC/AD era handling."""

    def test_bc(self) -> None:
        r = normalize_date("44 BC")
        assert r["year"] == -44
        assert r["era"] == "BC"

    def test_bce(self) -> None:
        r = normalize_date("44 BCE")
        assert r["year"] == -44
        assert r["era"] == "BC"

    def test_ad(self) -> None:
        r = normalize_date("476 AD")
        assert r["year"] == 476
        assert r["era"] == "AD"

    def test_ce(self) -> None:
        r = normalize_date("476 CE")
        assert r["year"] == 476
        assert r["era"] == "AD"

    def test_bc_full_date(self) -> None:
        r = normalize_date("March 15, 44 BC")
        assert r["year"] == -44
        assert r["month"] == 3
        assert r["day"] == 15
        assert r["era"] == "BC"

    def test_bc_iso_format(self) -> None:
        r = normalize_date("March 15, 44 BC")
        assert r["iso"] == "-0044-03-15"

    def test_default_era_is_ad(self) -> None:
        r = normalize_date("2024")
        assert r["era"] == "AD"


class TestNormalizeDateModifiers:
    """Circa, approximate, decades, ordinals."""

    def test_circa(self) -> None:
        r = normalize_date("circa 500")
        assert r["year"] == 500
        assert r["approximate"] is True

    def test_ca(self) -> None:
        r = normalize_date("ca. 1200")
        assert r["year"] == 1200
        assert r["approximate"] is True

    def test_approximately(self) -> None:
        r = normalize_date("approximately 1800")
        assert r["approximate"] is True

    def test_tilde(self) -> None:
        r = normalize_date("~1950")
        assert r["approximate"] is True
        assert r["year"] == 1950

    def test_decade(self) -> None:
        r = normalize_date("1800s")
        assert r["year"] == 1800
        assert r["precision"] == "decade"

    def test_ordinal_suffix(self) -> None:
        r = normalize_date("March 1st, 2024")
        assert r["day"] == 1
        assert r["month"] == 3

    def test_ordinal_15th(self) -> None:
        r = normalize_date("15th March 2024")
        assert r["day"] == 15


class TestNormalizeDateEdgeCases:
    """Edge cases and error handling."""

    def test_empty_string(self) -> None:
        r = normalize_date("")
        assert r["error"] is not None
        assert r["iso"] is None

    def test_whitespace_only(self) -> None:
        r = normalize_date("   ")
        assert r["error"] is not None

    def test_unparseable(self) -> None:
        r = normalize_date("yesterday")
        assert r["error"] is not None
        assert r["iso"] is None

    def test_garbage(self) -> None:
        r = normalize_date("not a date at all")
        assert r["error"] is not None

    def test_original_preserved(self) -> None:
        r = normalize_date("March 15, 44 BC")
        assert r["original"] == "March 15, 44 BC"


# ------------------------------------------------------------------ #
#  DateNormalizerTool tests (via ToolSpec)                            #
# ------------------------------------------------------------------ #


class TestDateNormalizerTool:
    """Test via execute()."""

    @pytest.fixture
    def tool(self) -> DateNormalizerTool:
        return DateNormalizerTool()

    def test_execute_basic(self, tool: DateNormalizerTool) -> None:
        result = tool.execute(date_string="2024-03-15")
        assert result["iso"] == "2024-03-15"
        assert result["error"] is None

    def test_execute_bc(self, tool: DateNormalizerTool) -> None:
        result = tool.execute(date_string="44 BC")
        assert result["year"] == -44

    def test_execute_error(self, tool: DateNormalizerTool) -> None:
        result = tool.execute(date_string="gobbledygook")
        assert result["error"] is not None

    def test_schema_rejects_missing(self) -> None:
        with pytest.raises(ValidationError):
            DateNormalizerArgs()  # type: ignore[call-arg]

    def test_is_toolspec(self) -> None:
        from workbench.tools.base import ToolSpec

        assert issubclass(DateNormalizerTool, ToolSpec)

    def test_name(self) -> None:
        assert DateNormalizerTool().name == "date_normalizer"

    def test_repr(self) -> None:
        assert "DateNormalizerTool" in repr(DateNormalizerTool())
