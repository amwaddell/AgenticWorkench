"""
Core workbench components.

This module provides the foundational types, interfaces, configuration,
and registry for building swappable RAG components.
"""

from workbench.core.config import (
    WorkbenchConfig,
    create_config_snapshot,
    load_config,
    merge_configs,
    save_config_snapshot,
)
from workbench.core.interfaces import (
    Chunker,
    ContextBuilder,
    Embedder,
    KeywordSearcher,
    LanguageModel,
    Reranker,
    Retriever,
    Tool,
    VectorSearcher,
)
from workbench.core.registry import (
    ComponentRegistry,
    get_registry,
    register_component,
)
from workbench.core.run_context import (
    RunContext,
    clear_run_context,
    generate_run_id,
    get_run_context,
    require_run_context,
    run_context,
    set_run_context,
)
from workbench.core.types import (
    Answer,
    Chunk,
    ChunkRef,
    Document,
    ModelResponse,
    Query,
    RerankResult,
    RetrievalResult,
    RunRecord,
)

__all__ = [
    # Types
    "Document",
    "Chunk",
    "ChunkRef",
    "Query",
    "RetrievalResult",
    "RerankResult",
    "ModelResponse",
    "Answer",
    "RunRecord",
    # Interfaces
    "LanguageModel",
    "Embedder",
    "KeywordSearcher",
    "VectorSearcher",
    "Reranker",
    "Chunker",
    "Tool",
    "ContextBuilder",
    "Retriever",
    # Config
    "WorkbenchConfig",
    "load_config",
    "merge_configs",
    "create_config_snapshot",
    "save_config_snapshot",
    # Registry
    "ComponentRegistry",
    "get_registry",
    "register_component",
    # Run Context
    "RunContext",
    "run_context",
    "get_run_context",
    "set_run_context",
    "clear_run_context",
    "require_run_context",
    "generate_run_id",
]
