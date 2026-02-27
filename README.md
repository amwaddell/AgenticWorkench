# Agentic RAG Workbench

A local-first retrieval-augmented generation (RAG) system built on **LangGraph**, designed for historical research over Wikipedia.  Everything runs on your machine: a local LLM via llama.cpp, local embeddings, LanceDB for vector + keyword search, and Phoenix for tracing.

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd agentic-workbench
pip install -e .
pip install -r requirements.txt
pip install -r requirements-dev.txt        # pytest, ruff

# 2. Download a Wikipedia dump (~300 MB)
bash scripts/download_wikipedia.sh

# 3. Ingest and build indexes
python scripts/ingest_wikipedia_dump.py
python scripts/build_vector_index.py
python scripts/build_keyword_index.py

# 4. Download and start the LLM server
bash scripts/start_model_server.sh         # needs llama.cpp installed

# 5. Run preflight checks
python scripts/preflight.py --no-phoenix

# 6. Launch the chat UI
bash scripts/run_ui.sh
```

Open **http://localhost:8501** — pick a graph from the sidebar and ask a question.

---

## Architecture

```
User question
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Supervisor Graph                               │
│  ┌───────────┐                                  │
│  │ Heuristic │──→ route ∈ {local_rag,           │
│  │ + LLM     │       web_research,              │
│  │ classifier│       timeline_research, math}   │
│  └───────────┘                                  │
│       │                                         │
│       ▼                                         │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ RAG Graph│  │ Researcher   │  │ Calculator│  │
│  │          │  │ Graph        │  │           │  │
│  └──────────┘  └──────────────┘  └───────────┘  │
└─────────────────────────────────────────────────┘
     │
     ▼
Answer + citations + trace spans
```

Three LangGraph workflows, composable via a supervisor:

| Graph | Topology | Use case |
|-------|----------|----------|
| `rag_graph` | search → open → answer (+ optional web fallback) | Factual Q&A |
| `researcher_graph` | search → open → timeline → answer → validate → repair | Historical research with citation discipline |
| `supervisor_graph` | route → specialist subgraph → merge | Auto-routing across all capabilities |

### Core Abstractions

**ChunkStore** (`stores/chunk_store.py`) — single source of truth for chunk data, backed by LanceDB.

**ToolSpec** (`tools/base.py`) — base class for all tools.  Provides Pydantic input validation, automatic tracing/logging/metrics, and a `.to_langchain_tool()` adapter for LangGraph interop.

**GraphRunner** (`systems/runner.py`) — unified entry point.  Builds components from config, compiles the requested graph, runs it with full observability, and returns the final state dict.

---

## Project Layout

```
├── configs/
│   └── defaults.yaml              # All tunables in one place
├── data/
│   ├── raw/wikipedia/             # Downloaded dump files
│   ├── processed/                 # Parquet articles
│   └── indexes/active/lancedb/    # Vector + keyword indexes
├── scripts/
│   ├── preflight.py               # Unified health check (replaces verify_day*.py)
│   ├── run_ui.sh                  # Launch Streamlit chat
│   ├── run_researcher_eval.py     # Evaluation harness
│   ├── start_model_server.sh      # llama.cpp server
│   └── download_wikipedia.sh      # Fetch Simple English Wikipedia
├── src/workbench/
│   ├── agents/                    # AgentState + shims to legacy
│   ├── core/                      # Config, run context, types
│   ├── data_build/                # Ingest, chunk, embed pipelines
│   ├── evaluation/                # Datasets, measures, reports
│   ├── graphs/                    # LangGraph workflows (primary orchestration)
│   │   ├── routing/               # Heuristic + LLM query classifier
│   │   ├── rag_graph.py
│   │   ├── researcher_graph.py
│   │   └── supervisor_graph.py
│   ├── legacy/                    # Retired AgentLoop + policies (do not extend)
│   ├── models/                    # LLM client (llama.cpp server)
│   ├── observability/             # Tracing, logging, metrics, Phoenix
│   ├── prompting/                 # Prompt templates, citation validation
│   ├── retrieval/                 # Hybrid search, reranking
│   ├── stores/                    # ChunkStore abstraction
│   ├── systems/                   # GraphRunner (recommended entry point)
│   ├── tools/                     # ToolSpec implementations
│   └── ui/                        # Streamlit chat app
├── tests/
├── runs/                          # Logs, traces, reports (gitignored)
├── docs/                          # Extension guides
│   ├── adding-tools.md
│   ├── adding-graphs.md
│   ├── observability.md
│   └── configuration.md
├── requirements.txt               # Core deps
├── requirements-dev.txt           # pytest, ruff
├── requirements-phoenix.txt       # Optional: Phoenix + LangChain instrumentation
└── requirements-models.txt        # Optional: llama-cpp-python
```

---

## Running Graphs

### Chat UI (recommended)

```bash
bash scripts/run_ui.sh
```

Select `rag_graph`, `researcher_graph`, or `supervisor_graph` from the sidebar.  Answers stream in real-time with a citations panel.

### CLI — single question

```python
from workbench.systems.runner import run_graph

result = run_graph("researcher_graph", "When did the Roman Republic end?")
print(result["answer_text"])
print(result["citations"])
print(result["timeline"])
```

### CLI — GraphRunner (reusable)

```python
from workbench.systems.runner import GraphRunner

runner = GraphRunner()                         # loads configs/defaults.yaml
result = runner.run("supervisor_graph", "What is 355/113?")
```

### Evaluation

```bash
# Run all 20 history questions through the researcher graph
python scripts/run_researcher_eval.py

# Single question
python scripts/run_researcher_eval.py --question-id hist_001

# Use the supervisor instead
python scripts/run_researcher_eval.py --system supervisor_graph
```

Reports are written to `runs/reports/<run_id>/`.

---

## Phoenix Tracing (Optional)

```bash
pip install -r requirements-phoenix.txt
phoenix serve                                  # http://localhost:6006
```

Then run any graph — spans are exported automatically.  The preflight script can start Phoenix for you:

```bash
python scripts/preflight.py                    # checks + starts Phoenix
```

Every tool call, LLM generation, retrieval, and graph node produces a span.

---

## Configuration

All settings live in `configs/defaults.yaml`.  Key sections:

| Section | Controls |
|---------|----------|
| `model` | LLM server URL, model name, temperature, max tokens |
| `embeddings` | Embedding model, device (mps/cpu/cuda), batch size |
| `retrieval` | Hybrid search parameters (top-k, RRF k, merge count) |
| `reranking` | Cross-encoder model and toggle |
| `web_search` | Provider (duckduckgo/tavily), confidence threshold |
| `supervisor` | Routing mode (hybrid/heuristic\_only/llm\_only) |
| `paths` | Data directories, index locations |
| `observability` | Tracing toggle, Phoenix endpoint, log level |

Override at runtime:

```python
runner = GraphRunner(config_overrides={"model": {"temperature": 0.3}})
```

See [docs/configuration.md](docs/configuration.md) for the full reference.

---

## Extending the Workbench

| Guide | What it covers |
|-------|----------------|
| [docs/adding-tools.md](docs/adding-tools.md) | Create a new ToolSpec, register it, wire it into a graph |
| [docs/adding-graphs.md](docs/adding-graphs.md) | Build a new LangGraph workflow, add it to the runner and UI |
| [docs/observability.md](docs/observability.md) | Tracing, logging, metrics, Phoenix setup |
| [docs/configuration.md](docs/configuration.md) | Full config reference, environment variables, overrides |

---

## Development

```bash
# Run tests (fast — no model server needed)
pytest

# Run integration tests (requires model server + indexes)
pytest -m integration

# Lint + format
ruff check src/ tests/ scripts/
ruff format src/ tests/ scripts/

# Preflight (verifies everything is wired correctly)
python scripts/preflight.py --no-phoenix --quick
```

---

## Migration from Legacy AgentLoop

The `AgentLoop` + policy classes have been moved to `workbench.legacy`.  Existing imports from `workbench.agents.loops` and `workbench.agents.policies` still work via thin shims, but new code should use the graph-based approach:

| Legacy | Graph equivalent |
|--------|-----------------|
| `AgentLoop` + `ScriptedPolicy` | `workbench.graphs.rag_graph` |
| `AgentLoop` + `ResearcherPolicy` | `workbench.graphs.researcher_graph` |
| (manual routing) | `workbench.graphs.supervisor_graph` |
| `ChatbotSystem` | `GraphRunner` / `run_graph()` |

The evaluation script (`scripts/run_researcher_eval.py`) has been ported to use `researcher_graph` directly.

---

## Requirements

- Python ≥ 3.11
- llama.cpp (`brew install llama.cpp` on macOS) or any OpenAI-compatible server
- ~2 GB disk for model weights + Wikipedia dump + indexes
- Apple Silicon recommended (Metal GPU acceleration) but CPU works too
