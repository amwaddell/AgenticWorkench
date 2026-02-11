#!/usr/bin/env python
"""
Simple examples showing how to use the model with and without run context.

Run this script to see both approaches in action.
"""

from workbench.core.config import create_config_snapshot, load_config
from workbench.core.run_context import run_context
from workbench.models import LlamaCppServerModel
from workbench.observability import setup_tracing, start_span

# Setup tracing (optional but recommended)
setup_tracing(service_name="simple-example")

print("=" * 70)
print("Example 1: Simple Usage (No Run Context)")
print("=" * 70)
print("\nThis is the easiest way - just create and use the model.\n")

# Load config
config = load_config()

# Create model
model = LlamaCppServerModel(
    base_url=config.model["base_url"],
    model_name=config.model["model_name"],
)

# Generate text
messages = [{"role": "user", "content": "Say hello in one sentence."}]

with start_span("simple_generation"):
    response = model.generate(messages, max_tokens=50)

print(f"Response: {response.text}")
print(f"Tokens: {response.tokens_out}, Latency: {response.latency_ms:.0f}ms")

print("\n✓ This works great for:")
print("  - Quick experiments")
print("  - Testing prompts")
print("  - Learning")
print("\n✗ But you won't get:")
print("  - Log files (runs/logs/)")
print("  - Metrics (runs/metrics/)")

print("\n" + "=" * 70)
print("Example 2: Full Production Usage (With Run Context)")
print("=" * 70)
print("\nThis gives you complete observability.\n")

# Create config snapshot
config_snapshot = create_config_snapshot(config)

# Use run context
with run_context(config_snapshot, run_type="example") as ctx:
    print(f"Run ID: {ctx.run_id}")
    print(f"Run type: {ctx.run_type}")
    print()

    # Create model (same as before)
    model2 = LlamaCppServerModel(
        base_url=config.model["base_url"],
        model_name=config.model["model_name"],
    )

    # Generate text (same as before)
    messages2 = [{"role": "user", "content": "Count to 3."}]

    with start_span("production_generation"):
        response2 = model2.generate(messages2, max_tokens=50)

    print(f"Response: {response2.text}")
    print(f"Tokens: {response2.tokens_out}, Latency: {response2.latency_ms:.0f}ms")

    print("\n✓ Complete observability:")
    print(f"  - Logs: runs/logs/{ctx.run_id}.jsonl")
    print("  - Metrics: runs/metrics/metrics.sqlite")
    print("  - Traces: Check Phoenix at http://localhost:6006")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("\nBoth approaches work perfectly!")
print("\nUse simple approach for:")
print("  - Jupyter notebooks")
print("  - Quick scripts")
print("  - Experimentation")
print("\nUse run context for:")
print("  - Production systems")
print("  - Evaluation runs")
print("  - Debugging issues")
print("  - Audit trails")
print("\nThe model code is identical - just add run context when you need it!")
print("=" * 70)
