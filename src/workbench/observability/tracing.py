"""
OpenTelemetry tracing infrastructure.

Provides span creation and management for distributed tracing.
Integrates with Phoenix and other OTLP-compatible backends.
"""

from contextlib import contextmanager
from typing import Any

# Use stub if OpenTelemetry is not installed
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
except ImportError:
    from workbench.observability.otel_stub import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        Resource,
        TracerProvider,
        trace,
    )

from workbench.core.run_context import get_run_context

# Global tracer instance
_tracer: Any | None = None
_tracer_provider: Any | None = None


def setup_tracing(
    service_name: str = "agentic-workbench",
    exporter: Any | None = None,
    console_export: bool = False,
    force_reset: bool = False,
) -> Any:
    """
    Set up OpenTelemetry tracing.

    Args:
        service_name: Name of the service for trace identification
        exporter: Optional span exporter (defaults to no export)
        console_export: If True, also export to console for debugging
        force_reset: If True, allow resetting the tracer provider (for tests)

    Returns:
        Configured tracer instance
    """
    global _tracer, _tracer_provider

    # If already set up and not forcing reset, return existing tracer
    if _tracer is not None and not force_reset:
        return _tracer

    # Create resource with service name
    resource = Resource.create({"service.name": service_name})

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Add exporters
    if exporter is not None:
        _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

    if console_export:
        _tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Set as global tracer provider (only if not already set, unless force_reset)
    try:
        trace.set_tracer_provider(_tracer_provider)
    except Exception:
        # If already set and we're forcing reset, that's ok
        if not force_reset:
            raise

    # Get tracer
    _tracer = trace.get_tracer(__name__)

    return _tracer


def get_tracer() -> Any:
    """
    Get the global tracer instance.

    Returns:
        Tracer instance

    Raises:
        RuntimeError: If tracing hasn't been set up
    """
    global _tracer
    if _tracer is None:
        # Auto-setup with no exporter for basic usage
        setup_tracing()
    return _tracer


def shutdown_tracing() -> None:
    """Shutdown tracing and flush any pending spans."""
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()


@contextmanager
def start_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: Any = None,
):
    """
    Create a new trace span as a context manager.

    Args:
        name: Name of the span
        attributes: Optional attributes to attach to span
        kind: Span kind (INTERNAL, CLIENT, SERVER, etc.)

    Yields:
        Span instance

    Example:
        with start_span("retrieve_chunks", {"query": "test", "top_k": 5}):
            # Your code here
            pass
    """
    tracer = get_tracer()

    # Set default kind if not provided
    if kind is None:
        kind = trace.SpanKind.INTERNAL

    # Add run_id from context if available
    span_attributes = {}
    run_ctx = get_run_context()
    if run_ctx is not None:
        span_attributes["run_id"] = run_ctx.run_id
        span_attributes["run_type"] = run_ctx.run_type

    # Add user-provided attributes (sanitized)
    if attributes:
        for key, value in attributes.items():
            span_attributes[key] = span_attribute_safe(value)

    with tracer.start_as_current_span(
        name, kind=kind, attributes=span_attributes
    ) as span:
        try:
            yield span
        except Exception as e:
            # Record exception in span
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@contextmanager
def start_run_span(run_id: str, run_type: str = "unknown"):
    """
    Create a top-level span for an entire run.

    Args:
        run_id: Unique run identifier
        run_type: Type of run (chatbot, evaluation, batch)

    Yields:
        Span instance

    Example:
        with start_run_span("20240209_143052_a1b2c3d4", "chatbot"):
            # All spans created here will be children of this run span
            pass
    """
    with start_span(
        f"run.{run_type}",
        attributes={
            "run_id": run_id,
            "run_type": run_type,
        },
        kind=trace.SpanKind.SERVER,
    ) as span:
        yield span


def span_attribute_safe(value: Any, max_length: int = 200) -> Any:
    """
    Make a value safe for use as a span attribute.

    Truncates long strings, converts complex objects to strings,
    and handles None values.

    Args:
        value: Value to make safe
        max_length: Maximum string length

    Returns:
        Safe value for span attribute
    """
    if value is None:
        return ""

    # Convert to string if not a primitive type
    if not isinstance(value, (str, int, float, bool)):
        value = str(value)

    # Truncate long strings
    if isinstance(value, str) and len(value) > max_length:
        return value[:max_length] + "..."

    return value


def add_span_attributes(attributes: dict[str, Any]) -> None:
    """
    Add attributes to the current span.

    Args:
        attributes: Dictionary of attributes to add
    """
    span = trace.get_current_span()
    if span is not None:
        for key, value in attributes.items():
            span.set_attribute(key, span_attribute_safe(value))


def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """
    Add an event to the current span.

    Args:
        name: Event name
        attributes: Optional event attributes
    """
    span = trace.get_current_span()
    if span is not None:
        safe_attributes = {}
        if attributes:
            safe_attributes = {k: span_attribute_safe(v) for k, v in attributes.items()}
        span.add_event(name, attributes=safe_attributes)
