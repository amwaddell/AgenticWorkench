"""
Agent tools for the agentic workbench.

All tools inherit from ``ToolSpec`` which provides:
    - Pydantic input validation via ``args_schema``
    - Automatic tracing, logging, and metrics in ``execute()``
    - LangChain/LangGraph interop via ``to_langchain_tool()``
"""

from workbench.tools.base import ToolSpec
from workbench.tools.open_chunk import OpenChunkArgs, OpenChunkTool
from workbench.tools.search_wikipedia import SearchWikipediaArgs, SearchWikipediaTool
from workbench.tools.timeline import TimelineArgs, TimelineTool

__all__ = [
    "ToolSpec",
    "SearchWikipediaTool",
    "SearchWikipediaArgs",
    "OpenChunkTool",
    "OpenChunkArgs",
    "TimelineTool",
    "TimelineArgs",
]
