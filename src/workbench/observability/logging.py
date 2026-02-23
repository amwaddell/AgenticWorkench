"""
Structured logging to JSON Lines format.

Each run gets its own log file: runs/logs/<run_id>.jsonl
Each line is a complete JSON object for easy parsing and analysis.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from workbench.core.run_context import get_run_context


class JSONLinesLogger:
    """
    Logger that writes structured JSON Lines to a file.

    Each event is a single line with complete JSON object.
    """

    def __init__(self, log_path: Path):
        """
        Initialize logger.

        Args:
            log_path: Path to log file
        """
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        level: str = "INFO",
    ) -> None:
        """
        Log an event to the JSON Lines file.

        Args:
            event_type: Type of event (e.g., "run_started", "component_finished")
            data: Event data dictionary
            level: Log level (INFO, WARNING, ERROR)
        """
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "level": level,
        }

        # Add run context if available
        run_ctx = get_run_context()
        if run_ctx is not None:
            event["run_id"] = run_ctx.run_id
            event["run_type"] = run_ctx.run_type

        # Add event data
        if data:
            event["data"] = data

        # Write as single JSON line
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def log_run_started(
        self, run_id: str, run_type: str, config: dict[str, Any]
    ) -> None:
        """Log that a run has started."""
        self.log_event(
            "run_started",
            {
                "run_id": run_id,
                "run_type": run_type,
                "config": config,
            },
        )

    def log_component_started(
        self, component_name: str, component_type: str, params: dict | None = None
    ) -> None:
        """Log that a component has started execution."""
        self.log_event(
            "component_started",
            {
                "component_name": component_name,
                "component_type": component_type,
                "params": params or {},
            },
        )

    def log_component_finished(
        self,
        component_name: str,
        component_type: str,
        duration_ms: float,
        result_summary: dict | None = None,
    ) -> None:
        """Log that a component has finished execution."""
        self.log_event(
            "component_finished",
            {
                "component_name": component_name,
                "component_type": component_type,
                "duration_ms": duration_ms,
                "result": result_summary or {},
            },
        )

    def log_error(
        self, error_type: str, error_message: str, context: dict | None = None
    ) -> None:
        """Log an error."""
        self.log_event(
            "error",
            {
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {},
            },
            level="ERROR",
        )

    def log_query(self, query: str, query_id: str | None = None) -> None:
        """Log a user query."""
        self.log_event(
            "query",
            {
                "query": query[:200],  # Truncate long queries
                "query_length": len(query),
                "query_id": query_id,
            },
        )

    def log_retrieval(
        self,
        query: str,
        num_results: int,
        retrieval_method: str,
        duration_ms: float,
    ) -> None:
        """Log a retrieval operation."""
        self.log_event(
            "retrieval",
            {
                "query": query[:100],
                "num_results": num_results,
                "retrieval_method": retrieval_method,
                "duration_ms": duration_ms,
            },
        )

    def log_model_call(
        self,
        model_name: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: float,
    ) -> None:
        """Log a model API call."""
        self.log_event(
            "model_call",
            {
                "model_name": model_name,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "duration_ms": duration_ms,
                "tokens_per_second": tokens_out / (duration_ms / 1000)
                if duration_ms > 0
                else 0,
            },
        )


# Global logger registry
_loggers: dict[str, JSONLinesLogger] = {}


def get_logger(
    run_id: str | None = None, logs_dir: Path | None = None
) -> JSONLinesLogger:
    """
    Get or create a logger for a specific run.

    Args:
        run_id: Run ID (uses current context if not provided)
        logs_dir: Directory for log files (defaults to runs/logs)

    Returns:
        JSONLinesLogger instance
    """
    # Get run_id from context if not provided
    if run_id is None:
        run_ctx = get_run_context()
        if run_ctx is None:
            raise RuntimeError("No run_id provided and no run context set")
        run_id = run_ctx.run_id

    # Get or create logger for this run
    if run_id not in _loggers:
        if logs_dir is None:
            logs_dir = Path("runs/logs")
        log_path = logs_dir / f"{run_id}.jsonl"
        _loggers[run_id] = JSONLinesLogger(log_path)

    return _loggers[run_id]


def log_event(
    event_type: str, data: dict[str, Any] | None = None, level: str = "INFO"
) -> None:
    """
    Convenience function to log an event using the current run's logger.

    Args:
        event_type: Type of event
        data: Event data
        level: Log level
    """
    logger = get_logger()
    logger.log_event(event_type, data, level)


def read_log_file(log_path: Path) -> list[dict[str, Any]]:
    """
    Read and parse a JSON Lines log file.

    Args:
        log_path: Path to log file

    Returns:
        List of parsed event dictionaries
    """
    events = []
    if not log_path.exists():
        return events

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    return events
