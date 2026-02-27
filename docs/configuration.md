# Configuration Reference

All configuration lives in `configs/defaults.yaml`.  The `load_config()` function reads this file and returns a `WorkbenchConfig` object.

## Loading Config

```python
from workbench.core.config import load_config

# Default: reads configs/defaults.yaml
cfg = load_config()

# Custom file
cfg = load_config(config_path=Path("configs/my_config.yaml"))

# With overrides (merged on top of file)
cfg = load_config(overrides={"model": {"temperature": 0.3}})
```

## Full Reference

### `model` — LLM Server

```yaml
model:
  provider: "llamacpp_server"
  base_url: "http://localhost:8080"    # llama.cpp server URL
  model_name: "ministral-3-8b-instruct"
  temperature: 0.7                     # 0.0 = deterministic, 1.0 = creative
  max_tokens: 2048                     # max output tokens per generation
  timeout_seconds: 60                  # HTTP timeout for model calls
```

The workbench talks to any OpenAI-compatible API.  To use a remote provider, change `base_url` (e.g., to an Ollama or vLLM endpoint).

### `embeddings` — Embedding Model

```yaml
embeddings:
  provider: "sentence_transformers"
  model_name: "intfloat/multilingual-e5-base"
  batch_size: 32
  device: "mps"                        # "mps" (Apple Silicon), "cuda", or "cpu"
```

The embedding model runs in-process.  First run downloads weights from HuggingFace (~500 MB).

### `retrieval` — Hybrid Search

```yaml
retrieval:
  strategy: "hybrid"
  keyword_top_k: 20         # candidates from BM25 / FTS
  vector_top_k: 20          # candidates from vector search
  merge_top_n: 50           # candidates passed to RRF merge
  rrf_k: 60                 # RRF smoothing parameter
  final_top_k: 5            # chunks returned to the graph
```

### `reranking` — Cross-Encoder

```yaml
reranking:
  enabled: true
  provider: "cross_encoder"
  model_name: "BAAI/bge-reranker-v2-m3"
  top_n: 5                   # keep top N after reranking
```

Set `enabled: false` to skip reranking (faster but less precise).

### `context` — Context Window

```yaml
context:
  max_tokens: 2048           # max evidence tokens in the prompt
  max_chunks_per_doc: 3      # deduplicate chunks from same article
```

### `chunking` — Text Chunking

```yaml
chunking:
  strategy: "heading_aware"
  max_chunk_tokens: 512
  overlap_tokens: 50
  min_chunk_tokens: 100
```

### `web_search` — Web Fallback (Day 5)

```yaml
web_search:
  enabled: true
  provider: "duckduckgo"     # "duckduckgo" (zero-key) or "tavily"
  max_results: 5
  confidence_threshold: 2    # min retrieval results before web fallback
```

For Tavily, set the `TAVILY_API_KEY` environment variable and change provider to `"tavily"`.

### `source_fetch` — URL Extraction

```yaml
source_fetch:
  timeout_seconds: 15
  max_chars: 8000
  user_agent: "AgenticWorkbench/1.0"
```

### `supervisor` — Routing (Day 7)

```yaml
supervisor:
  routing_mode: "hybrid"     # "hybrid" | "heuristic_only" | "llm_only"
```

| Mode | Behavior |
|------|----------|
| `hybrid` | Heuristic first; falls back to LLM if confidence is low |
| `heuristic_only` | Pattern-matching only (fast, no LLM call) |
| `llm_only` | Always asks the LLM to classify (slower, more accurate) |

### `paths` — Data Directories

```yaml
paths:
  data_dir: "./data"
  raw_data: "./data/raw"
  processed_data: "./data/processed"
  indexes_dir: "./data/indexes"
  active_index: "./data/indexes/active"
  index_snapshots: "./data/indexes/snapshots"
```

### `runs` — Output Directories

```yaml
runs:
  output_dir: "./runs"
  logs_dir: "./runs/logs"
  traces_dir: "./runs/traces"
  reports_dir: "./runs/reports"
```

### `observability` — Tracing / Metrics

```yaml
observability:
  enable_tracing: true
  enable_metrics: true
  phoenix_endpoint: "http://localhost:6006"
  log_level: "INFO"
```

## Config Snapshots

Before each run, the config is frozen into an immutable snapshot for reproducibility:

```python
from workbench.core.config import create_config_snapshot

snapshot = create_config_snapshot(cfg)
# snapshot is a plain dict, safe to serialize to JSON
```

This snapshot is saved alongside run logs and reports.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `TAVILY_API_KEY` | Required if using `provider: "tavily"` for web search |
| `WORKBENCH_CONFIG` | Override the default config file path |
