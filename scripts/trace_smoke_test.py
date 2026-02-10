#!/usr/bin/env python
"""
Smoke test for observability infrastructure.

Tests that tracing, logging, and metrics all work together.
Sends traces to Phoenix if running, otherwise uses in-memory exporter.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workbench.core.run_context import run_context
from workbench.observability import (
    get_logger,
    get_metrics_store,
    is_phoenix_available,
    setup_phoenix_tracing,
    setup_tracing,
    start_run_span,
    start_span,
)


def simulate_component(name: str, duration_ms: float) -> str:
    """Simulate a component doing work."""
    with start_span(f"component.{name}", {"duration_target": duration_ms}):
        # Log component start
        logger = get_logger()
        logger.log_component_started(name, "test_component", {"duration": duration_ms})

        # Simulate work
        time.sleep(duration_ms / 1000)

        # Log component finish
        logger.log_component_finished(name, "test_component", duration_ms)

        # Record metrics
        metrics = get_metrics_store()
        metrics.record_component_latency(
            name, duration_ms, component_type="test_component"
        )

        return f"{name} completed in {duration_ms}ms"


def main():
    """Run smoke test."""
    print("=" * 70)
    print("Observability Smoke Test")
    print("=" * 70)

    # Check if Phoenix is available
    phoenix_available = is_phoenix_available()
    print(f"\n✓ Phoenix available: {phoenix_available}")

    # Setup tracing
    if phoenix_available:
        print("  Setting up Phoenix tracing...")
        setup_phoenix_tracing(console_export=True)
    else:
        print("  Phoenix not available, using basic tracing")
        print("  To use Phoenix:")
        print("    1. pip install arize-phoenix arize-phoenix-otel")
        print("    2. phoenix serve")
        setup_tracing(console_export=True)

    # Create run context
    config = {"test": "smoke_test", "version": "1.0"}

    with run_context(config, run_type="smoke_test") as ctx:
        print(f"\n✓ Run ID: {ctx.run_id}")
        print(f"  Log file: runs/logs/{ctx.run_id}.jsonl")
        print("  Metrics: runs/metrics/metrics.sqlite")

        # Get logger and log run start
        logger = get_logger()
        logger.log_run_started(ctx.run_id, "smoke_test", config)

        # Create top-level run span
        with start_run_span(ctx.run_id, "smoke_test"):
            print("\n✓ Creating nested spans...")

            # Level 1: Outer operation
            with start_span("outer_operation", {"operation": "test"}):
                result1 = simulate_component("retrieval", 100)
                print(f"  - {result1}")

                # Level 2: Nested operation
                with start_span("nested_operation", {"nested": True}):
                    result2 = simulate_component("reranking", 50)
                    print(f"  - {result2}")

            # Another top-level operation
            result3 = simulate_component("generation", 200)
            print(f"  - {result3}")

        # Log some events
        print("\n✓ Logging events...")
        logger.log_event("test_event", {"message": "This is a test event"})
        logger.log_query("What is the meaning of life?")
        logger.log_model_call(
            "test-model", tokens_in=100, tokens_out=50, duration_ms=250
        )

        # Record some metrics
        print("\n✓ Recording metrics...")
        metrics = get_metrics_store()
        metrics.record_model_tokens("test-model", 100, 50, 250)
        metrics.record_retrieval(
            final_count=5,
            query="test query",
            keyword_candidates=10,
            vector_candidates=10,
            reranked=5,
            duration_ms=150,
        )

    # Verify outputs
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    # Check log file
    log_file = Path(f"runs/logs/{ctx.run_id}.jsonl")
    if log_file.exists():
        with open(log_file) as f:
            log_lines = f.readlines()
        print(f"\n✓ Log file created: {len(log_lines)} events")
    else:
        print(f"\n✗ Log file not found: {log_file}")

    # Check metrics database
    db_file = Path("runs/metrics/metrics.sqlite")
    if db_file.exists():
        summary = metrics.get_run_summary(ctx.run_id)
        print("\n✓ Metrics database created")
        print(f"  - Component calls: {summary['component_calls']}")
        print(f"  - Total tokens in: {summary['total_tokens_in']}")
        print(f"  - Total tokens out: {summary['total_tokens_out']}")
        print(f"  - Retrieval calls: {summary['retrieval_calls']}")
    else:
        print(f"\n✗ Metrics database not found: {db_file}")

    # Phoenix instructions
    if phoenix_available:
        print("\n" + "=" * 70)
        print("View Traces in Phoenix")
        print("=" * 70)
        print("\nOpen your browser to: http://localhost:6006")
        print(f"Look for run_id: {ctx.run_id}")
    else:
        print("\n" + "=" * 70)
        print("To view traces in Phoenix, install and start it:")
        print("=" * 70)
        print("\n  pip install arize-phoenix arize-phoenix-otel")
        print("  phoenix serve")

    print("\n" + "=" * 70)
    print("✓ Smoke test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
