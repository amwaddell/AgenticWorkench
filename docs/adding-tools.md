# Adding Tools

This guide walks through creating a new tool, registering it, and wiring it into a graph.

## 1. Create the Tool Class

Every tool inherits from `ToolSpec` and defines a Pydantic `args_schema`.

```python
# src/workbench/tools/my_tool.py

from pydantic import BaseModel, Field
from workbench.tools.base import ToolSpec


class MyToolArgs(BaseModel):
    """Input schema — validated before execute() runs."""
    query: str = Field(..., description="The search query")
    limit: int = Field(default=5, ge=1, le=50)


class MyTool(ToolSpec):
    name = "my_tool"
    description = "Does something useful with a query."
    args_schema = MyToolArgs

    def __init__(self, some_dependency):
        self._dep = some_dependency

    def execute(self, **kwargs) -> dict:
        """
        Core logic.  ToolSpec handles:
        - Pydantic validation of kwargs against args_schema
        - Wrapping the call in a trace span
        - Logging + metrics recording

        Returns a plain dict (the graph reads from this).
        """
        args = MyToolArgs(**kwargs)
        result = self._dep.do_work(args.query, args.limit)
        return {"items": result, "count": len(result)}
```

Key rules:
- `execute()` receives `**kwargs`, not a Pydantic model — validate inside.
- Return a plain `dict`.  Graph nodes read specific keys from it.
- Keep the tool stateless if possible; inject dependencies via `__init__`.

## 2. Export from the Tools Package

Add your tool to `src/workbench/tools/__init__.py`:

```python
from workbench.tools.my_tool import MyTool, MyToolArgs

__all__ = [
    # ... existing tools ...
    "MyTool",
    "MyToolArgs",
]
```

## 3. Wire into a Graph

Graph nodes call `tool.execute(**params)` and write results into state.  Here's a minimal node factory:

```python
def _make_my_node(my_tool):
    def my_node(state):
        result = my_tool.execute(query=state["question"], limit=10)
        return {"my_data": result["items"]}
    return my_node
```

Then register it in your graph builder:

```python
builder.add_node("my_step", _make_my_node(my_tool))
builder.add_edge("previous_step", "my_step")
builder.add_edge("my_step", "next_step")
```

## 4. LangChain Interop

If you need the tool as a LangChain `BaseTool` (e.g., for a LangChain agent):

```python
lc_tool = my_tool.to_langchain_tool()
```

This is already handled by the `ToolSpec` base class.

## 5. Testing

Write a unit test that doesn't require external services:

```python
# tests/test_my_tool.py

def test_my_tool_basic():
    class FakeDep:
        def do_work(self, query, limit):
            return [{"id": 1}]

    tool = MyTool(some_dependency=FakeDep())
    result = tool.execute(query="test", limit=1)
    assert result["count"] == 1
    assert len(result["items"]) == 1


def test_my_tool_validates_input():
    tool = MyTool(some_dependency=None)
    # Missing required field should raise
    import pytest
    with pytest.raises(Exception):
        tool.execute(limit=5)  # no query
```

## Existing Tools Reference

| Tool | Module | Purpose |
|------|--------|---------|
| `SearchWikipediaTool` | `tools/search_wikipedia.py` | Hybrid search over LanceDB |
| `OpenChunkTool` | `tools/open_chunk.py` | Fetch full chunk text by ID |
| `TimelineTool` | `tools/timeline.py` | Extract chronological timeline from evidence |
| `WebSearchTool` | `tools/web_search.py` | DuckDuckGo / Tavily web search |
| `SourceFetchTool` | `tools/source_fetch.py` | Fetch + extract text from a URL |
| `CalculatorTool` | `tools/utility/calculator.py` | Safe math expression evaluator |
| `DateNormalizerTool` | `tools/utility/dates.py` | Normalize date strings |
