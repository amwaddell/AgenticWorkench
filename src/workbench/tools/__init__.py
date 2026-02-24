"""
Agent tools for the agentic workbench.

All tools inherit from ``ToolSpec`` which provides:
    - Pydantic input validation via ``args_schema``
    - Automatic tracing, logging, and metrics in ``execute()``
    - LangChain/LangGraph interop via ``to_langchain_tool()``

Core tools (Days 1–2):
    SearchWikipediaTool, OpenChunkTool, TimelineTool

Day 5 additions:
    WebSearchTool, SourceFetchTool, CalculatorTool, DateNormalizerTool
"""

from workbench.tools.base import ToolSpec
from workbench.tools.open_chunk import OpenChunkArgs, OpenChunkTool
from workbench.tools.search_wikipedia import SearchWikipediaArgs, SearchWikipediaTool
from workbench.tools.source_fetch import SourceFetchArgs, SourceFetchTool
from workbench.tools.timeline import TimelineArgs, TimelineTool
from workbench.tools.utility.calculator import CalculatorArgs, CalculatorTool
from workbench.tools.utility.dates import DateNormalizerArgs, DateNormalizerTool
from workbench.tools.web_search import WebSearchArgs, WebSearchTool

__all__ = [
    # Base
    "ToolSpec",
    # Core tools
    "SearchWikipediaTool",
    "SearchWikipediaArgs",
    "OpenChunkTool",
    "OpenChunkArgs",
    "TimelineTool",
    "TimelineArgs",
    # Day 5 tools
    "WebSearchTool",
    "WebSearchArgs",
    "SourceFetchTool",
    "SourceFetchArgs",
    "CalculatorTool",
    "CalculatorArgs",
    "DateNormalizerTool",
    "DateNormalizerArgs",
]
