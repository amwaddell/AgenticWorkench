"""
Tests for tracing infrastructure.

Uses in-memory span exporter so tests don't depend on Phoenix.
"""

try:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    USING_REAL_OTEL = True
except ImportError:
    from workbench.observability.otel_stub import (
        InMemorySpanExporter,
        SimpleSpanProcessor,
    )

    USING_REAL_OTEL = False

from workbench.core.run_context import run_context
from workbench.observability.tracing import (
    add_span_attributes,
    add_span_event,
    span_attribute_safe,
    start_run_span,
    start_span,
)

# Global exporter and tracer provider for all tests
_test_exporter = None
_test_tracer_provider = None


def _setup_test_tracing():
    """Set up tracing for tests with an in-memory exporter."""
    global _test_exporter, _test_tracer_provider

    if _test_exporter is None:
        _test_exporter = InMemorySpanExporter()

        if USING_REAL_OTEL:
            # Use real OpenTelemetry setup
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider

            resource = Resource.create({"service.name": "test-service"})
            _test_tracer_provider = TracerProvider(resource=resource)
            # Use SimpleSpanProcessor for immediate export (no batching)
            _test_tracer_provider.add_span_processor(
                SimpleSpanProcessor(_test_exporter)
            )

            # Set as global provider (only once)
            try:
                trace.set_tracer_provider(_test_tracer_provider)
            except Exception:
                # Already set, that's ok
                pass
        else:
            # Use stub setup
            from workbench.observability.tracing import setup_tracing

            setup_tracing("test-service", exporter=_test_exporter)

    # Clear spans before each test
    _test_exporter.clear()
    return _test_exporter


def test_setup_tracing():
    """Test that tracing can be set up."""
    exporter = _setup_test_tracing()
    assert exporter is not None


def test_span_creation_with_in_memory_exporter():
    """Test that spans are created and exported to in-memory exporter."""
    exporter = _setup_test_tracing()

    # Create a span
    with start_span("test_operation", {"test_attr": "test_value"}):
        pass

    # Get exported spans
    spans = exporter.get_finished_spans()

    # Should have one span
    assert len(spans) >= 1, f"Expected spans but got {len(spans)}"

    # Find our test span
    test_span = next((s for s in spans if s.name == "test_operation"), None)
    assert test_span is not None, f"Test span not found in {[s.name for s in spans]}"
    assert test_span.attributes.get("test_attr") == "test_value"


def test_nested_spans():
    """Test that nested spans maintain parent-child relationships."""
    exporter = _setup_test_tracing()

    with start_span("parent_operation"):
        with start_span("child_operation"):
            pass

    spans = exporter.get_finished_spans()

    # Should have at least 2 spans
    assert len(spans) >= 2, f"Expected 2+ spans but got {len(spans)}"

    # Find parent and child
    parent = next((s for s in spans if s.name == "parent_operation"), None)
    child = next((s for s in spans if s.name == "child_operation"), None)

    assert parent is not None, "Parent span not found"
    assert child is not None, "Child span not found"

    # Child should have parent's span ID as its parent
    assert child.parent is not None, "Child has no parent"
    assert child.parent.span_id == parent.context.span_id


def test_run_span_with_context():
    """Test run span with run context."""
    exporter = _setup_test_tracing()

    config = {"test": "config"}

    with run_context(config, run_type="test") as ctx:
        with start_run_span(ctx.run_id, "test"):
            pass

    spans = exporter.get_finished_spans()

    # Find run span
    run_span = next((s for s in spans if "run.test" in s.name), None)
    assert run_span is not None, f"Run span not found in {[s.name for s in spans]}"
    assert run_span.attributes.get("run_id") == ctx.run_id
    assert run_span.attributes.get("run_type") == "test"


def test_span_attributes_from_context():
    """Test that run context attributes are automatically added to spans."""
    exporter = _setup_test_tracing()

    config = {"test": "config"}

    with run_context(config, run_type="test") as ctx:
        with start_span("operation_with_context"):
            pass

    spans = exporter.get_finished_spans()
    span = next((s for s in spans if s.name == "operation_with_context"), None)

    assert span is not None, "Span not found"
    assert span.attributes.get("run_id") == ctx.run_id
    assert span.attributes.get("run_type") == "test"


def test_add_span_attributes():
    """Test adding attributes to current span."""
    exporter = _setup_test_tracing()

    with start_span("test_operation"):
        add_span_attributes({"dynamic_attr": "value", "count": 42})

    spans = exporter.get_finished_spans()
    span = next((s for s in spans if s.name == "test_operation"), None)

    assert span is not None, "Span not found"
    assert span.attributes.get("dynamic_attr") == "value"
    assert span.attributes.get("count") == 42


def test_add_span_event():
    """Test adding events to current span."""
    exporter = _setup_test_tracing()

    with start_span("test_operation"):
        add_span_event("something_happened", {"detail": "test"})

    spans = exporter.get_finished_spans()
    span = next((s for s in spans if s.name == "test_operation"), None)

    assert span is not None, "Span not found"
    assert len(span.events) > 0, "No events found"

    # Find our event
    event = next((e for e in span.events if e.name == "something_happened"), None)
    assert event is not None, "Event not found"


def test_span_attribute_safe_truncates_long_strings():
    """Test that long strings are truncated."""
    long_string = "a" * 500
    safe_value = span_attribute_safe(long_string, max_length=100)

    assert isinstance(safe_value, str)
    assert len(safe_value) <= 103  # 100 + "..."
    assert safe_value.endswith("...")


def test_span_attribute_safe_handles_none():
    """Test that None values are converted to empty string."""
    assert span_attribute_safe(None) == ""


def test_span_attribute_safe_converts_objects():
    """Test that complex objects are converted to strings."""
    obj = {"key": "value", "nested": {"data": 123}}
    safe_value = span_attribute_safe(obj)

    assert isinstance(safe_value, str)
    assert "key" in safe_value


def test_span_exception_recording():
    """Test that exceptions are recorded in spans."""
    exporter = _setup_test_tracing()

    try:
        with start_span("failing_operation"):
            raise ValueError("Test error")
    except ValueError:
        pass  # Expected

    spans = exporter.get_finished_spans()
    span = next((s for s in spans if s.name == "failing_operation"), None)

    assert span is not None, "Span not found"
    assert len(span.events) > 0, "No events found"

    # Should have exception event
    exception_events = [e for e in span.events if "exception" in e.name.lower()]
    assert len(exception_events) > 0, "No exception events found"
