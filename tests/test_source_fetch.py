"""
Tests for the SourceFetchTool (Day 5).

Uses ``unittest.mock.patch`` to mock httpx responses so tests
don't hit the network.  Validates:

- HTML text extraction (via BeautifulSoup)
- Plain text pass-through
- Non-text content handling
- HTTP error handling
- Timeout handling
- max_chars truncation
- ToolSpec integration
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from workbench.tools.source_fetch import (
    SourceFetchArgs,
    SourceFetchTool,
    _extract_text_bs4,
)

# ------------------------------------------------------------------ #
#  HTML extraction unit tests                                         #
# ------------------------------------------------------------------ #

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<nav>Navigation links here</nav>
<article>
<h1>Main Article</h1>
<p>This is the first paragraph with important content.</p>
<p>This is the second paragraph.</p>
</article>
<script>var x = 1;</script>
<footer>Footer content</footer>
</body>
</html>
"""


class TestExtractTextBs4:
    """Test HTML → text extraction."""

    def test_extracts_article_text(self) -> None:
        text = _extract_text_bs4(SAMPLE_HTML, max_chars=5000)
        assert "Main Article" in text
        assert "first paragraph" in text
        assert "second paragraph" in text

    def test_removes_script_tags(self) -> None:
        text = _extract_text_bs4(SAMPLE_HTML, max_chars=5000)
        assert "var x = 1" not in text

    def test_removes_nav_tags(self) -> None:
        text = _extract_text_bs4(SAMPLE_HTML, max_chars=5000)
        assert "Navigation links" not in text

    def test_removes_footer_tags(self) -> None:
        text = _extract_text_bs4(SAMPLE_HTML, max_chars=5000)
        assert "Footer content" not in text

    def test_respects_max_chars(self) -> None:
        text = _extract_text_bs4(SAMPLE_HTML, max_chars=20)
        assert len(text) <= 20

    def test_empty_html(self) -> None:
        text = _extract_text_bs4("", max_chars=5000)
        assert text == ""

    def test_plain_text_in_html(self) -> None:
        text = _extract_text_bs4("<p>Just text</p>", max_chars=5000)
        assert "Just text" in text


# ------------------------------------------------------------------ #
#  Schema tests                                                       #
# ------------------------------------------------------------------ #


class TestSourceFetchSchema:
    """Verify Pydantic schema validation."""

    def test_valid_args(self) -> None:
        args = SourceFetchArgs(url="https://example.com")
        assert args.url == "https://example.com"
        assert args.max_chars == 8000  # default

    def test_custom_max_chars(self) -> None:
        args = SourceFetchArgs(url="https://example.com", max_chars=1000)
        assert args.max_chars == 1000

    def test_missing_url_raises(self) -> None:
        with pytest.raises(ValidationError):
            SourceFetchArgs()  # type: ignore[call-arg]

    def test_max_chars_too_low(self) -> None:
        with pytest.raises(ValidationError):
            SourceFetchArgs(url="https://example.com", max_chars=10)

    def test_max_chars_too_high(self) -> None:
        with pytest.raises(ValidationError):
            SourceFetchArgs(url="https://example.com", max_chars=100000)


# ------------------------------------------------------------------ #
#  Execution tests (mocked HTTP)                                      #
# ------------------------------------------------------------------ #


def _make_mock_response(
    text: str = "",
    content_type: str = "text/html; charset=utf-8",
    status_code: int = 200,
) -> MagicMock:
    """Create a mock httpx Response."""
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx

        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


class TestSourceFetchExecution:
    """Test execute() with mocked HTTP."""

    @pytest.fixture
    def tool(self) -> SourceFetchTool:
        return SourceFetchTool(timeout=5.0)

    @patch("workbench.tools.source_fetch.httpx.get")
    def test_fetch_html_success(
        self, mock_get: MagicMock, tool: SourceFetchTool
    ) -> None:
        mock_get.return_value = _make_mock_response(
            text=SAMPLE_HTML, content_type="text/html"
        )
        result = tool.execute(url="https://example.com/article")
        assert result["fetched"] is True
        assert result["error"] is None
        assert "Main Article" in result["text"]
        assert result["char_count"] > 0

    @patch("workbench.tools.source_fetch.httpx.get")
    def test_fetch_plain_text(self, mock_get: MagicMock, tool: SourceFetchTool) -> None:
        mock_get.return_value = _make_mock_response(
            text="Just plain text here.", content_type="text/plain"
        )
        result = tool.execute(url="https://example.com/readme.txt")
        assert result["fetched"] is True
        assert result["text"] == "Just plain text here."

    @patch("workbench.tools.source_fetch.httpx.get")
    def test_fetch_non_text(self, mock_get: MagicMock, tool: SourceFetchTool) -> None:
        mock_get.return_value = _make_mock_response(
            text="", content_type="application/pdf"
        )
        result = tool.execute(url="https://example.com/doc.pdf")
        assert result["fetched"] is True
        assert "Non-text content" in result["text"]

    @patch("workbench.tools.source_fetch.httpx.get")
    def test_http_error(self, mock_get: MagicMock, tool: SourceFetchTool) -> None:
        mock_get.return_value = _make_mock_response(status_code=404)
        result = tool.execute(url="https://example.com/missing")
        assert result["fetched"] is False
        assert result["error"] is not None
        assert "404" in result["error"]

    @patch("workbench.tools.source_fetch.httpx.get")
    def test_timeout_error(self, mock_get: MagicMock, tool: SourceFetchTool) -> None:
        import httpx

        mock_get.side_effect = httpx.TimeoutException("timed out")
        result = tool.execute(url="https://example.com/slow")
        assert result["fetched"] is False
        assert "TimeoutException" in result["error"]

    @patch("workbench.tools.source_fetch.httpx.get")
    def test_connection_error(self, mock_get: MagicMock, tool: SourceFetchTool) -> None:
        import httpx

        mock_get.side_effect = httpx.ConnectError("connection refused")
        result = tool.execute(url="https://example.com/down")
        assert result["fetched"] is False
        assert result["error"] is not None

    @patch("workbench.tools.source_fetch.httpx.get")
    def test_max_chars_respected(
        self, mock_get: MagicMock, tool: SourceFetchTool
    ) -> None:
        long_html = "<p>" + "A" * 20000 + "</p>"
        mock_get.return_value = _make_mock_response(
            text=long_html, content_type="text/html"
        )
        result = tool.execute(url="https://example.com/long", max_chars=500)
        assert result["char_count"] <= 500

    @patch("workbench.tools.source_fetch.httpx.get")
    def test_url_in_result(self, mock_get: MagicMock, tool: SourceFetchTool) -> None:
        mock_get.return_value = _make_mock_response(text="<p>Hi</p>")
        result = tool.execute(url="https://example.com/test")
        assert result["url"] == "https://example.com/test"


# ------------------------------------------------------------------ #
#  ToolSpec integration                                               #
# ------------------------------------------------------------------ #


class TestSourceFetchToolSpec:
    """Verify ToolSpec base class integration."""

    def test_is_toolspec(self) -> None:
        from workbench.tools.base import ToolSpec

        assert issubclass(SourceFetchTool, ToolSpec)

    def test_name(self) -> None:
        assert SourceFetchTool().name == "source_fetch"

    def test_has_description(self) -> None:
        assert len(SourceFetchTool().description) > 10

    def test_repr(self) -> None:
        assert "SourceFetchTool" in repr(SourceFetchTool())
