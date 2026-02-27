# Observability

The workbench has three observability layers: tracing (OpenTelemetry → Phoenix), structured logging (JSONL), and metrics (SQLite).  All three are opt-in and degrade gracefully when disabled.

## Tracing (OpenTelemetry + Phoenix)

### Setup

```bash
pip install -r requirements-phoenix.txt
phoenix serve                              # http://localhost:6006
```

Tracing is enabled by default in `configs/defaults.yaml`:

```yaml
observability:
  enable_tracing: true
  phoenix_endpoint: "http://localhost:6006"
```

### What Gets Traced

Every tool call, LLM generation, retrieval, and graph node produces a span:

```
researcher_graph.run
├── graph.retrieve_node
│   └── tool.search_wikipedia
├── graph.open_node
│   └── tool.open_chunk (×N)
├── graph.timeline_node
│   └── tool.timeline
├── graph.answer_node
│   └── model.generate
└── graph.validate_node
```

### Using Spans in Your Code

```python
from workbench.observability.tracing import start_span, add_span_attributes

with start_span("my_operation", attributes={"key": "value"}):
    result = do_work()
    add_span_attributes({"result_count": len(result)})
```

### LangGraph Auto-Instrumentation

If `openinference-instrumentation-langchain` is installed (included in `requirements-phoenix.txt`), LangGraph node transitions are automatically traced in Phoenix without any extra code.

## Logging (JSONL)

Every run produces a structured JSONL log file at `runs/logs/<run_id>.jsonl`.

### Log Events

| Event | When |
|-------|------|
| `run_started` | Run context opens |
| `query` | Question received |
| `retrieval` | Search completed |
| `model_call` | LLM generation completed |
| `agent_completed` | Run finished |

### Using the Logger

```python
from workbench.observability.logging import get_logger

logger = get_logger(run_id="my-run-123")
logger.log_query("What is photosynthesis?")
logger.log_event("custom_event", {"key": "value"})
```

Logs are written to `runs/logs/` by default (configurable via `runs.logs_dir` in config).

## Metrics (SQLite)

Latency, token counts, and retrieval stats are recorded in `runs/metrics/metrics.sqlite`.

### Recording Metrics

```python
from workbench.observability.metrics import get_metrics_store

metrics = get_metrics_store()

metrics.record_component_latency(
    component_name="my_component",
    duration_ms=150.5,
    component_type="tool",
)

metrics.record_model_tokens(
    model_name="ministral-3-8b",
    tokens_in=500,
    tokens_out=200,
    duration_ms=3200.0,
)

metrics.record_retrieval(
    final_count=5,
    query="test query",
    keyword_candidates=20,
    vector_candidates=20,
    reranked=10,
    duration_ms=85.0,
)
```

### Querying Metrics

The SQLite database can be queried directly:

```bash
sqlite3 runs/metrics/metrics.sqlite "SELECT * FROM component_latency ORDER BY timestamp DESC LIMIT 10;"
```

## Run Context

Every execution is wrapped in a `run_context` that provides a unique `run_id` and coordinates logging, tracing, and metrics:

```python
from workbench.core.run_context import run_context

with run_context(config_snapshot, run_type="evaluation") as ctx:
    print(ctx.run_id)      # e.g. "20250225-143052-a1b2c3"
    print(ctx.run_type)    # "evaluation"
    # All logging/tracing/metrics within this block use this run_id
```

The `GraphRunner` handles this automatically — you only need `run_context` if building custom pipelines.

## Disabling Observability

Set in `configs/defaults.yaml`:

```yaml
observability:
  enable_tracing: false
  enable_metrics: false
```

Or at the code level, the tracing functions become no-ops when Phoenix is not running.  Logging always writes to disk (it's lightweight).
