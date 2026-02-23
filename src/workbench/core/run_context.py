"""
Run context management using contextvars.

This allows any component to access the current run context
without passing it through every function call.
"""

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional

# Context variable for the current run
_run_context: ContextVar[Optional["RunContext"]] = ContextVar(
    "run_context", default=None
)


@dataclass
class RunContext:
    """
    Context for a single run (experiment, evaluation, or chatbot session).

    Contains metadata and configuration that should be accessible
    throughout the run without explicit passing.
    """

    run_id: str
    start_time: datetime
    config_snapshot: dict[str, Any]
    run_type: str = "unknown"  # chatbot, evaluation, batch
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        config_snapshot: dict[str, Any],
        run_type: str = "unknown",
        run_id: str | None = None,
        **metadata: Any,
    ) -> "RunContext":
        """
        Create a new run context.

        Args:
            config_snapshot: Frozen configuration for this run
            run_type: Type of run (chatbot, evaluation, batch)
            run_id: Optional run ID (generates UUID if not provided)
            **metadata: Additional metadata to store

        Returns:
            New RunContext instance
        """
        if run_id is None:
            run_id = generate_run_id()

        return cls(
            run_id=run_id,
            start_time=datetime.now(UTC),
            config_snapshot=config_snapshot,
            run_type=run_type,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "run_type": self.run_type,
            "config_snapshot": self.config_snapshot,
            "metadata": self.metadata,
        }


def generate_run_id() -> str:
    """
    Generate a unique run ID.

    Format: timestamp_uuid
    Example: 20240209_143052_a1b2c3d4
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


def set_run_context(context: RunContext) -> None:
    """
    Set the current run context.

    Args:
        context: RunContext to set as current
    """
    _run_context.set(context)


def get_run_context() -> RunContext | None:
    """
    Get the current run context.

    Returns:
        Current RunContext or None if not set
    """
    return _run_context.get()


def clear_run_context() -> None:
    """Clear the current run context."""
    _run_context.set(None)


def require_run_context() -> RunContext:
    """
    Get the current run context, raising an error if not set.

    Returns:
        Current RunContext

    Raises:
        RuntimeError: If no run context is set
    """
    context = get_run_context()
    if context is None:
        raise RuntimeError(
            "No run context set. Use set_run_context() or run within a run_context() manager."
        )
    return context


class run_context:
    """
    Context manager for setting run context.

    Usage:
        with run_context(config_snapshot, run_type="chatbot") as ctx:
            # Code here has access to run context
            current = get_run_context()
    """

    def __init__(
        self,
        config_snapshot: dict[str, Any],
        run_type: str = "unknown",
        run_id: str | None = None,
        **metadata: Any,
    ):
        """
        Initialize run context manager.

        Args:
            config_snapshot: Frozen configuration
            run_type: Type of run
            run_id: Optional run ID
            **metadata: Additional metadata
        """
        self.context = RunContext.create(
            config_snapshot=config_snapshot,
            run_type=run_type,
            run_id=run_id,
            **metadata,
        )
        self.token = None

    def __enter__(self) -> RunContext:
        """Enter context and set run context."""
        self.token = _run_context.set(self.context)
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous run context."""
        if self.token is not None:
            _run_context.reset(self.token)
        return False
