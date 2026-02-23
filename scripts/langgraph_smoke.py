"""
LangGraph smoke test — Day 0.

Defines a tiny StateGraph, compiles it, invokes it once,
and optionally sends a trace to Phoenix.

Usage:
    python scripts/langgraph_smoke.py
    PHOENIX_TRACE=1 python scripts/langgraph_smoke.py   # with Phoenix tracing

Prerequisites:
    pip install -r requirements.txt
    (optional) pip install -r requirements-phoenix.txt && phoenix serve
"""

from __future__ import annotations

import os
import sys
from typing import TypedDict

from langgraph.graph import END, StateGraph

# ── tiny state & nodes ──────────────────────────────────────────────


class SmallState(TypedDict):
    message: str
    steps: int


def greet(state: SmallState) -> SmallState:
    """First node: set a greeting."""
    return {
        "message": f"Hello from LangGraph! (input was: {state['message']})",
        "steps": state["steps"] + 1,
    }


def shout(state: SmallState) -> SmallState:
    """Second node: upper-case the message."""
    return {"message": state["message"].upper(), "steps": state["steps"] + 1}


# ── graph construction ──────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Build and compile a two-node StateGraph."""
    graph = StateGraph(SmallState)
    graph.add_node("greet", greet)
    graph.add_node("shout", shout)

    graph.set_entry_point("greet")
    graph.add_edge("greet", "shout")
    graph.add_edge("shout", END)

    return graph.compile()


# ── Phoenix auto-instrumentation (opt-in) ───────────────────────────


def _maybe_enable_phoenix_tracing() -> bool:
    """
    If PHOENIX_TRACE=1 is set, wire up OpenInference auto-instrumentation
    for LangChain/LangGraph so every invoke() produces a trace in Phoenix.

    Returns True if instrumentation was activated.
    """
    if os.getenv("PHOENIX_TRACE", "").strip() not in ("1", "true", "yes"):
        return False

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        from workbench.observability.phoenix import setup_phoenix_tracing

        # Set up the base OTel pipeline pointing at Phoenix
        setup_phoenix_tracing(service_name="langgraph-smoke")

        # Layer LangChain auto-instrumentation on top
        LangChainInstrumentor().instrument()

        print("✓ Phoenix LangChain/LangGraph auto-instrumentation enabled")
        return True

    except ImportError as exc:
        print(
            f"⚠  Could not enable Phoenix tracing (missing package: {exc}).\n"
            "   Install with: pip install -r requirements-phoenix.txt"
        )
        return False


# ── main ────────────────────────────────────────────────────────────


def main() -> None:
    print("─── LangGraph smoke test ───")

    # Optional: Phoenix tracing
    phoenix_on = _maybe_enable_phoenix_tracing()

    # Build & invoke
    app = build_graph()
    result = app.invoke({"message": "workbench", "steps": 0})

    # Report
    print(f"  result message : {result['message']}")
    print(f"  result steps   : {result['steps']}")

    assert result["steps"] == 2, f"Expected 2 steps, got {result['steps']}"
    assert result["message"] == "HELLO FROM LANGGRAPH! (INPUT WAS: WORKBENCH)"

    print("✓ LangGraph smoke test passed")

    if phoenix_on:
        # Give the batch exporter a moment to flush
        from workbench.observability.tracing import shutdown_tracing

        shutdown_tracing()
        print("✓ Traces flushed to Phoenix — check http://localhost:6006")
    else:
        print("  (set PHOENIX_TRACE=1 to send traces to Phoenix)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"✗ LangGraph smoke test FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
