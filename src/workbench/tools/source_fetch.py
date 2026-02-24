"""
Source fetch tool: fetch a URL and extract readable text.

Given a URL (typically from a web search result), this tool:

1. Fetches the page with ``httpx``
2. Strips HTML to readable text with ``BeautifulSoup``
3. Truncates to a configurable character limit

This is the "read more" action for web search results, analogous
to ``OpenChunkTool`` for Wikipedia chunks.

Usage (direct)::

    tool = SourceFetchTool()
    result = tool.execute(url="https://example.com/article")

Usage (LangGraph)::

    lc_tool = tool.to_langchain_tool()
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from pydantic import BaseModel, Field

from workbench.observability.tracing import add_span_attributes
from workbench.tools.base import ToolSpec

# Default character limit for extracted text
_DEFAULT_MAX_CHARS = 8000
_DEFAULT_TIMEOUT = 15.0


# ------------------------------------------------------------------ #
#  Args schema                                                        #
# ------------------------------------------------------------------ #


class SourceFetchArgs(BaseModel):
    """Input schema for the source_fetch tool."""

    url: str = Field(..., description="The URL to fetch and extract text from.")
    max_chars: int = Field(
        default=_DEFAULT_MAX_CHARS,
        ge=100,
        le=50000,
        description="Maximum characters of extracted text to return.",
    )


# ------------------------------------------------------------------ #
#  Text extraction helpers                                            #
# ------------------------------------------------------------------ #


def _extract_text_bs4(html: str, max_chars: int) -> str:
    """Extract readable text from HTML using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "beautifulsoup4 is required for the source_fetch tool. "
            "Install it with: pip install beautifulsoup4"
        ) from exc

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Extract text
    text = soup.get_text(separator="\n", strip=True)

    # Collapse excessive whitespace / blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text[:max_chars].strip()


# ------------------------------------------------------------------ #
#  Tool implementation                                                #
# ------------------------------------------------------------------ #


class SourceFetchTool(ToolSpec):
    """
    Fetch a URL and extract its readable text content.

    Args:
        timeout: HTTP request timeout in seconds.
        user_agent: User-Agent header for requests.
    """

    name: str = "source_fetch"
    description: str = (
        "Fetch a web page by URL and extract its readable text content. "
        "Use this after web_search to read full articles."
    )
    args_schema = SourceFetchArgs

    def __init__(
        self,
        timeout: float = _DEFAULT_TIMEOUT,
        user_agent: str = "AgenticWorkbench/1.0",
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Fetch and extract text from a URL.

        Args (validated by SourceFetchArgs):
            url: The URL to fetch.
            max_chars: Maximum characters of text to return.

        Returns:
            Dict with keys: url, text, char_count, fetched (bool),
            error (str or None).
        """
        url: str = kwargs["url"]
        max_chars: int = kwargs["max_chars"]

        try:
            response = httpx.get(
                url,
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            # Only extract from HTML; for plain text, return as-is
            if "text/html" in content_type:
                text = _extract_text_bs4(response.text, max_chars)
            elif "text/plain" in content_type:
                text = response.text[:max_chars]
            else:
                text = f"[Non-text content: {content_type}]"

            add_span_attributes(
                {
                    "url": url[:200],
                    "char_count": len(text),
                    "content_type": content_type[:100],
                }
            )

            return {
                "url": url,
                "text": text,
                "char_count": len(text),
                "fetched": True,
                "error": None,
            }

        except httpx.HTTPStatusError as exc:
            return {
                "url": url,
                "text": "",
                "char_count": 0,
                "fetched": False,
                "error": f"HTTP {exc.response.status_code}: {str(exc)[:200]}",
            }
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            return {
                "url": url,
                "text": "",
                "char_count": 0,
                "fetched": False,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }

    def __repr__(self) -> str:
        return f"SourceFetchTool(timeout={self.timeout})"
