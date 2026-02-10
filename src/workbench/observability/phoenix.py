"""
Arize Phoenix integration for trace visualization.

Phoenix is a local trace viewer that runs on port 6006.
Start it with: phoenix serve
"""

import os

# Use stub if OpenTelemetry is not installed
try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ImportError:
    from workbench.observability.otel_stub import OTLPSpanExporter

from workbench.observability.tracing import setup_tracing


def get_phoenix_endpoint() -> str:
    """
    Get Phoenix OTLP endpoint from environment or default.

    Environment variable: PHOENIX_ENDPOINT
    Default: http://localhost:6006/v1/traces

    Returns:
        Phoenix endpoint URL
    """
    return os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")


def setup_phoenix_tracing(
    service_name: str = "agentic-workbench",
    endpoint: str | None = None,
    console_export: bool = False,
):
    """
    Set up tracing with Phoenix as the backend.

    Args:
        service_name: Service name for traces
        endpoint: Optional custom Phoenix endpoint
        console_export: If True, also print traces to console

    Returns:
        Configured tracer

    Example:
        from workbench.observability.phoenix import setup_phoenix_tracing

        setup_phoenix_tracing()

        with start_span("my_operation"):
            # Your code here
            pass
    """
    if endpoint is None:
        endpoint = get_phoenix_endpoint()

    # Create OTLP exporter pointing to Phoenix
    exporter = OTLPSpanExporter(endpoint=endpoint)

    # Setup tracing with Phoenix exporter
    tracer = setup_tracing(
        service_name=service_name,
        exporter=exporter,
        console_export=console_export,
    )

    print(f"✓ Tracing configured to send to Phoenix at {endpoint}")
    print("  View traces at: http://localhost:6006")

    return tracer


def is_phoenix_available(endpoint: str | None = None) -> bool:
    """
    Check if Phoenix is available at the given endpoint.

    Args:
        endpoint: Optional endpoint to check (uses default if None)

    Returns:
        True if Phoenix is reachable, False otherwise
    """
    if endpoint is None:
        endpoint = get_phoenix_endpoint()

    try:
        import httpx

        # Try to connect to Phoenix base URL
        base_url = endpoint.replace("/v1/traces", "")
        response = httpx.get(base_url, timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def phoenix_instructions() -> str:
    """
    Get instructions for starting Phoenix.

    Returns:
        Help text for starting Phoenix
    """
    return """
To start Phoenix locally:

1. Install Phoenix:
   pip install arize-phoenix arize-phoenix-otel

2. Start the server:
   phoenix serve

3. Open browser to:
   http://localhost:6006

Phoenix will now receive and visualize traces from your workbench.
"""
