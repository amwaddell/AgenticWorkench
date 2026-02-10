"""
Minimal OpenTelemetry stubs for testing when OTel is not installed.

This allows tests to run without requiring OpenTelemetry installation.
In production, install the real packages:
  pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
"""

# Check if real OpenTelemetry is available
try:
    from opentelemetry import trace as _real_trace
    from opentelemetry.sdk.resources import Resource as _real_Resource
    from opentelemetry.sdk.trace import TracerProvider as _real_TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor as _real_BatchSpanProcessor,
    )
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter as _real_ConsoleSpanExporter,
    )
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor as _real_SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter as _real_InMemorySpanExporter,
    )

    # Re-export real implementations
    trace = _real_trace
    Resource = _real_Resource
    TracerProvider = _real_TracerProvider
    BatchSpanProcessor = _real_BatchSpanProcessor
    ConsoleSpanExporter = _real_ConsoleSpanExporter
    SimpleSpanProcessor = _real_SimpleSpanProcessor
    InMemorySpanExporter = _real_InMemorySpanExporter

    OTEL_AVAILABLE = True

except ImportError:
    # Provide stub implementations
    OTEL_AVAILABLE = False

    # Global span storage
    _current_span = None
    _span_stack = []

    class StubEvent:
        """Stub event."""

        def __init__(self, name, attributes=None):
            self.name = name
            self.attributes = attributes or {}

    class StubSpanContext:
        """Stub span context."""

        def __init__(self):
            import random

            self.span_id = f"span_{random.randint(1000, 9999)}"
            self.trace_id = f"trace_{random.randint(1000, 9999)}"

    class StubSpan:
        """Stub span that tracks calls."""

        def __init__(self, name="", attributes=None, parent=None):
            self.name = name
            self.attributes = dict(attributes) if attributes else {}
            self.events = []
            self.context = StubSpanContext()
            # Store parent context for compatibility
            if parent and hasattr(parent, "context"):
                self.parent = parent.context
            else:
                self.parent = parent

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def add_event(self, name, attributes=None):
            event = StubEvent(name, attributes)
            self.events.append(event)

        def set_status(self, status):
            pass

        def record_exception(self, exception):
            self.add_event("exception", {"exception.type": type(exception).__name__})

        def __enter__(self):
            global _current_span, _span_stack
            _span_stack.append(_current_span)
            _current_span = self
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            global _current_span, _span_stack
            if _span_stack:
                _current_span = _span_stack.pop()
            else:
                _current_span = None
            return False

    class StubTracer:
        """Stub tracer that creates stub spans."""

        def __init__(self):
            self.spans = []

        def start_as_current_span(self, name, kind=None, attributes=None):
            global _current_span
            parent = _current_span
            span = StubSpan(name=name, attributes=attributes, parent=parent)
            self.spans.append(span)
            return span

    class StubTracerProvider:
        """Stub tracer provider."""

        def __init__(self, resource=None):
            self.resource = resource
            self.processors = []
            self.tracer = StubTracer()

        def add_span_processor(self, processor):
            self.processors.append(processor)
            # If processor has an exporter with a spans list, give it access
            if hasattr(processor, "exporter") and hasattr(processor.exporter, "spans"):
                # Share the tracer's span list with the exporter
                processor.exporter.spans = self.tracer.spans

        def get_tracer(self, name):
            return self.tracer

        def shutdown(self):
            pass

    class trace:
        """Stub trace module."""

        SpanKind = type(
            "SpanKind",
            (),
            {
                "INTERNAL": 1,
                "CLIENT": 2,
                "SERVER": 3,
                "PRODUCER": 4,
                "CONSUMER": 5,
            },
        )()

        class Status:
            """Stub Status class."""

            def __init__(self, code, description=""):
                self.code = code
                self.description = description

        StatusCode = type("StatusCode", (), {"ERROR": 2, "OK": 1})()

        _tracer_provider = None

        @staticmethod
        def set_tracer_provider(provider):
            trace._tracer_provider = provider

        @staticmethod
        def get_tracer(name):
            if trace._tracer_provider:
                return trace._tracer_provider.get_tracer(name)
            return StubTracer()

        @staticmethod
        def get_current_span():
            global _current_span
            if _current_span is None:
                return StubSpan()
            return _current_span

    class Resource:
        """Stub resource."""

        def __init__(self, attributes=None):
            self.attributes = attributes or {}

        @staticmethod
        def create(attributes):
            return Resource(attributes)

    class TracerProvider(StubTracerProvider):
        pass

    class BatchSpanProcessor:
        """Stub batch span processor."""

        def __init__(self, exporter):
            self.exporter = exporter

    class ConsoleSpanExporter:
        """Stub console exporter."""

        pass

    class SimpleSpanProcessor:
        """Stub simple span processor."""

        def __init__(self, exporter):
            self.exporter = exporter

    class InMemorySpanExporter:
        """Stub in-memory exporter that collects spans."""

        def __init__(self):
            self.spans = []

        def get_finished_spans(self):
            return self.spans

        def clear(self):
            self.spans = []


# Check for OTLP exporter
try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as _real_OTLPSpanExporter,
    )

    OTLPSpanExporter = _real_OTLPSpanExporter
    OTLP_AVAILABLE = True

except ImportError:
    OTLP_AVAILABLE = False

    class OTLPSpanExporter:
        """Stub OTLP exporter."""

        def __init__(self, endpoint=None, **kwargs):
            self.endpoint = endpoint
            self.spans = []
