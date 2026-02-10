"""
Observability components for the agentic workbench.

Provides tracing, structured logging, and metrics collection.
"""

from workbench.observability.logging import (
    JSONLinesLogger,
    get_logger,
    log_event,
    read_log_file,
)
from workbench.observability.metrics import (
    MetricsStore,
    get_metrics_store,
    record_component_latency,
)
from workbench.observability.phoenix import (
    get_phoenix_endpoint,
    is_phoenix_available,
    phoenix_instructions,
    setup_phoenix_tracing,
)
from workbench.observability.tracing import (
    add_span_attributes,
    add_span_event,
    get_tracer,
    setup_tracing,
    shutdown_tracing,
    span_attribute_safe,
    start_run_span,
    start_span,
)

__all__ = [
    # Tracing
    "setup_tracing",
    "get_tracer",
    "shutdown_tracing",
    "start_span",
    "start_run_span",
    "add_span_attributes",
    "add_span_event",
    "span_attribute_safe",
    # Phoenix
    "setup_phoenix_tracing",
    "get_phoenix_endpoint",
    "is_phoenix_available",
    "phoenix_instructions",
    # Logging
    "JSONLinesLogger",
    "get_logger",
    "log_event",
    "read_log_file",
    # Metrics
    "MetricsStore",
    "get_metrics_store",
    "record_component_latency",
]
