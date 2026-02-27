# Adding Graphs

This guide covers building a new LangGraph workflow, registering it with the runner, and making it available in the UI.

## 1. Define the State

Every graph needs a TypedDict state schema.  Extend from the common fields or create your own:

```python
# src/workbench/graphs/my_graph.py

from typing import Any, TypedDict


class MyGraphState(TypedDict, total=False):
    # Input
    question: str
    search_top_k: int
    open_top_n: int

    # Pipeline stages
    retrieved: list[dict[str, Any]]
    opened: list[dict[str, Any]]
    my_custom_data: list[dict[str, Any]]   # your new stage

    # Output
    answer_text: str
    citations: list[str]
    tokens_in: int
    tokens_out: int
    model_latency_ms: float
    error: str | None
```

Use `total=False` so nodes only need to return the keys they update.

## 2. Write Node Functions

Each node receives the full state and returns a partial update dict:

```python
def _make_my_custom_node(my_tool):
    def my_custom_node(state: MyGraphState) -> dict[str, Any]:
        if state.get("error"):
            return {}  # skip on prior error

        result = my_tool.execute(query=state["question"])
        return {"my_custom_data": result["items"]}

    return my_custom_node
```

Patterns to follow:
- Wrap work in `start_span(...)` for tracing.
- Return `{}` on error states to avoid cascading failures.
- Use `add_span_attributes(...)` for metrics.

## 3. Build and Compile the Graph

```python
from langgraph.graph import END, StateGraph


def build_my_graph(
    search_tool,
    open_chunk_tool,
    my_tool,
    model,
    **kwargs,
):
    builder = StateGraph(MyGraphState)

    # Register nodes
    builder.add_node("retrieve", _make_retrieve_node(search_tool))
    builder.add_node("open", _make_open_node(open_chunk_tool))
    builder.add_node("my_step", _make_my_custom_node(my_tool))
    builder.add_node("answer", _make_answer_node(model))

    # Wire edges
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "open")
    builder.add_edge("open", "my_step")
    builder.add_edge("my_step", "answer")
    builder.add_edge("answer", END)

    return builder.compile()
```

For conditional branching:

```python
def _should_branch(state: MyGraphState) -> str:
    if some_condition(state):
        return "branch_a"
    return "branch_b"

builder.add_conditional_edges(
    "my_step",
    _should_branch,
    {"branch_a": "node_a", "branch_b": "node_b"},
)
```

## 4. Add a Convenience Runner (Optional)

Follow the pattern in `researcher_graph.py`:

```python
def run_my_graph(graph, question, config_snapshot=None, **kwargs):
    """Invoke with full observability."""
    from workbench.core.run_context import run_context
    from workbench.observability.tracing import setup_tracing

    setup_tracing(service_name="my-graph")
    with run_context(config_snapshot or {}, run_type="my_graph") as ctx:
        result = graph.invoke({"question": question, **kwargs})
        return result
```

## 5. Register with the Unified Runner

Edit `src/workbench/systems/runner.py`:

```python
# Add to the imports
from workbench.graphs.my_graph import build_my_graph

# Add to SUPPORTED_GRAPHS
_CUSTOM_GRAPHS = {"my_graph"}
SUPPORTED_GRAPHS = _BASIC_GRAPHS | _TIMELINE_GRAPHS | _FULL_GRAPHS | _CUSTOM_GRAPHS

# Add a branch in build_graph()
if graph_name == "my_graph":
    return build_my_graph(
        search_tool=c["search_tool"],
        open_chunk_tool=c["open_chunk_tool"],
        my_tool=c.get("my_tool"),
        model=c["model"],
    )
```

Now `GraphRunner().run("my_graph", "...")` works.

## 6. Add to the Streamlit UI

Edit `src/workbench/ui/streamlit_app.py` — add your graph name to the sidebar selector list:

```python
graph_name = st.sidebar.selectbox(
    "Graph",
    ["supervisor_graph", "researcher_graph", "rag_graph", "my_graph"],
)
```

## 7. Export from the Graphs Package

Update `src/workbench/graphs/__init__.py`:

```python
from workbench.graphs.my_graph import build_my_graph, MyGraphState

__all__ = [
    # ... existing exports ...
    "build_my_graph",
    "MyGraphState",
]
```

## 8. Testing

Test compilation with fakes (no external services):

```python
def test_my_graph_compiles():
    class FakeSearch:
        def execute(self, **kw): return {"chunks": []}

    class FakeOpen:
        def execute(self, **kw): return {"found": False, "chunk": None}

    class FakeModel:
        def generate(self, msgs, **kw):
            from workbench.core.types import ModelResponse
            return ModelResponse(text="Answer.", tokens_in=1, tokens_out=1, latency_ms=1)

    graph = build_my_graph(
        search_tool=FakeSearch(),
        open_chunk_tool=FakeOpen(),
        my_tool=FakeMyTool(),
        model=FakeModel(),
    )
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "stream")
```

## Existing Graphs Reference

| Graph | Entry point | Nodes | Key feature |
|-------|-------------|-------|-------------|
| `rag_graph` | `build_rag_graph()` | retrieve → open → answer | Web search fallback |
| `researcher_graph` | `build_researcher_graph()` | retrieve → open → timeline → answer → validate → repair | Citation validation loop |
| `supervisor_graph` | `build_supervisor_graph()` | route → specialist → merge | Heuristic + LLM routing |
