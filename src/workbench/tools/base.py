"""
Tool framework: ToolSpec base class with schema validation and adapters.

Every tool in the workbench inherits from ``ToolSpec``.  This gives you:

    1. **Pydantic args validation** — each tool declares an ``args_schema``
       (a Pydantic BaseModel) so callers get clear errors on bad input.
    2. **Built-in observability** — every ``execute()`` call automatically
       creates a trace span, emits a JSONL log event, and records a metric.
    3. **LangChain / LangGraph interop** — call ``to_langchain_tool()`` to
       get a ``StructuredTool`` that LangGraph's ``ToolNode`` can execute.

Subclasses implement ``run(**kwargs) -> dict`` with the actual logic.
The ``execute()`` wrapper handles validation + hooks so ``run()`` stays clean.

Usage (direct)::

    tool = SearchWikipediaTool(retriever=my_retriever)
    result = tool.execute(query="French Revolution", top_k=5)

Usage (LangGraph)::

    lc_tool = tool.to_langchain_tool()
    # Pass to ToolNode, bind to ChatModel, etc.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from workbench.observability.tracing import add_span_attributes, start_span


class ToolSpec(ABC):
    """
    Base class for all workbench tools.

    Subclasses must set:
        name:        Short machine-readable tool name (e.g. "search_wikipedia").
        description: One-line summary for LLM tool-use prompts.
        args_schema: A Pydantic BaseModel class describing accepted kwargs.

    Subclasses must implement:
        run(**kwargs) -> dict[str, Any]
    """

    name: str
    description: str
    args_schema: type[BaseModel]

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Validate inputs, run the tool, and record observability hooks.

        This is the primary entry point for the agent loop.  It wraps
        ``run()`` with:

        * Pydantic arg validation (raises ``ValidationError`` on bad input)
        * A trace span named ``tool.<name>``
        * A JSONL log event (best-effort — no error if logging unavailable)
        * A component-latency metric (best-effort)

        Returns:
            dict produced by ``run()``.

        Raises:
            pydantic.ValidationError: If kwargs don't match ``args_schema``.
        """
        # --- 1. validate ------------------------------------------------
        validated: BaseModel = self.args_schema(**kwargs)
        validated_kwargs = validated.model_dump()

        # --- 2. execute inside a span -----------------------------------
        span_name = f"tool.{self.name}"

        # Pick a few representative attributes for the span (not the full payload)
        span_attrs = self._pick_span_attributes(validated_kwargs)

        with start_span(span_name, attributes=span_attrs):
            t0 = time.time()
            try:
                result = self.run(**validated_kwargs)
                duration_ms = (time.time() - t0) * 1000

                add_span_attributes(
                    {
                        "duration_ms": round(duration_ms, 1),
                        "status": "success",
                    }
                )

                # --- 3. observability (best-effort) ----------------------
                self._log_event(validated_kwargs, result, duration_ms, status="success")
                self._record_metric(duration_ms, status="success")

                return result

            except Exception as exc:
                duration_ms = (time.time() - t0) * 1000

                add_span_attributes(
                    {
                        "duration_ms": round(duration_ms, 1),
                        "status": "error",
                        "error": str(exc)[:200],
                    }
                )

                self._log_event(
                    validated_kwargs, {}, duration_ms, status="error", error=str(exc)
                )
                self._record_metric(duration_ms, status="error")
                raise

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the tool logic.  Subclasses implement this.

        Args are guaranteed to have passed ``args_schema`` validation
        before reaching this method.

        Returns:
            Result dictionary.
        """
        ...

    def to_langchain_tool(self) -> Any:
        """
        Return a LangChain ``StructuredTool`` for use in LangGraph ToolNode.

        The wrapper calls ``self.execute()`` (with full validation and hooks)
        and serialises the result dict to a JSON string, which is the format
        LangGraph's ``ToolNode`` expects.

        Raises:
            ImportError: If ``langchain_core`` is not installed.
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as exc:
            raise ImportError(
                "langchain-core is required for to_langchain_tool(). "
                "Install it with: pip install langchain-core"
            ) from exc

        # Capture self so the closure can call execute()
        tool_instance = self

        def _run(**kwargs: Any) -> str:
            result = tool_instance.execute(**kwargs)
            return json.dumps(result, default=str)

        return StructuredTool(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            func=_run,
        )

    # ------------------------------------------------------------------ #
    #  Observability helpers (best-effort — never fail the tool call)     #
    # ------------------------------------------------------------------ #

    def _pick_span_attributes(
        self, kwargs: dict[str, Any], max_keys: int = 5
    ) -> dict[str, Any]:
        """Select a small subset of kwargs as span attributes."""
        attrs: dict[str, Any] = {"tool_name": self.name}
        for key in list(kwargs)[:max_keys]:
            val = kwargs[key]
            if isinstance(val, (str, int, float, bool)):
                # Truncate long strings
                if isinstance(val, str) and len(val) > 200:
                    val = val[:200] + "..."
                attrs[key] = val
            elif isinstance(val, list):
                attrs[f"{key}_count"] = len(val)
        return attrs

    def _log_event(
        self,
        params: dict[str, Any],
        result: dict[str, Any],
        duration_ms: float,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        """Emit a JSONL log event for this tool call (best-effort)."""
        try:
            from workbench.observability.logging import log_event

            data: dict[str, Any] = {
                "tool_name": self.name,
                "duration_ms": round(duration_ms, 1),
                "status": status,
            }
            # Add a compact param summary (avoid huge payloads)
            param_summary = {
                k: (v if isinstance(v, (str, int, float, bool)) else type(v).__name__)
                for k, v in list(params.items())[:6]
            }
            data["params"] = param_summary

            if error:
                data["error"] = error[:300]

            log_event("tool_executed", data)
        except Exception:
            # Logging should never break a tool call
            pass

    def _record_metric(self, duration_ms: float, status: str = "success") -> None:
        """Record component latency metric (best-effort)."""
        try:
            from workbench.observability.metrics import record_component_latency

            record_component_latency(
                component_name=f"tool.{self.name}",
                duration_ms=duration_ms,
                component_type="tool",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                     #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
