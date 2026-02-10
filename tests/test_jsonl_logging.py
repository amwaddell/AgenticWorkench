"""
Tests for JSON Lines logging.
"""

import tempfile
from pathlib import Path

from workbench.core.run_context import run_context
from workbench.observability.logging import (
    JSONLinesLogger,
    get_logger,
    log_event,
    read_log_file,
)


def test_jsonl_logger_creates_file():
    """Test that logger creates a log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLinesLogger(log_path)

        logger.log_event("test_event", {"key": "value"})

        assert log_path.exists()


def test_jsonl_logger_writes_valid_json():
    """Test that logger writes valid JSON lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLinesLogger(log_path)

        logger.log_event("event1", {"data": "value1"})
        logger.log_event("event2", {"data": "value2"})

        # Read and parse
        events = read_log_file(log_path)

        assert len(events) == 2
        assert events[0]["event_type"] == "event1"
        assert events[0]["data"]["data"] == "value1"
        assert events[1]["event_type"] == "event2"


def test_jsonl_logger_includes_required_fields():
    """Test that events include required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLinesLogger(log_path)

        logger.log_event("test_event", {"key": "value"})

        events = read_log_file(log_path)
        event = events[0]

        # Required fields
        assert "timestamp" in event
        assert "event_type" in event
        assert "level" in event
        assert event["event_type"] == "test_event"
        assert event["level"] == "INFO"


def test_jsonl_logger_includes_run_context():
    """Test that events include run context when available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLinesLogger(log_path)

        config = {"test": "config"}

        with run_context(config, run_type="test") as ctx:
            logger.log_event("test_event", {"key": "value"})

        events = read_log_file(log_path)
        event = events[0]

        assert "run_id" in event
        assert "run_type" in event
        assert event["run_id"] == ctx.run_id
        assert event["run_type"] == "test"


def test_log_run_started():
    """Test logging run start event."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLinesLogger(log_path)

        logger.log_run_started("test_run_123", "test", {"key": "value"})

        events = read_log_file(log_path)
        event = events[0]

        assert event["event_type"] == "run_started"
        assert event["data"]["run_id"] == "test_run_123"
        assert event["data"]["run_type"] == "test"
        assert event["data"]["config"]["key"] == "value"


def test_log_component_started_and_finished():
    """Test logging component lifecycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLinesLogger(log_path)

        logger.log_component_started("retriever", "retriever", {"top_k": 5})
        logger.log_component_finished(
            "retriever", "retriever", 123.45, {"num_results": 5}
        )

        events = read_log_file(log_path)

        assert len(events) == 2
        assert events[0]["event_type"] == "component_started"
        assert events[0]["data"]["component_name"] == "retriever"
        assert events[0]["data"]["params"]["top_k"] == 5

        assert events[1]["event_type"] == "component_finished"
        assert events[1]["data"]["duration_ms"] == 123.45
        assert events[1]["data"]["result"]["num_results"] == 5


def test_log_error():
    """Test logging errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLinesLogger(log_path)

        logger.log_error(
            "ValueError", "Something went wrong", {"component": "retriever"}
        )

        events = read_log_file(log_path)
        event = events[0]

        assert event["event_type"] == "error"
        assert event["level"] == "ERROR"
        assert event["data"]["error_type"] == "ValueError"
        assert event["data"]["error_message"] == "Something went wrong"


def test_log_query():
    """Test logging queries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLinesLogger(log_path)

        logger.log_query("What is the meaning of life?", query_id="q123")

        events = read_log_file(log_path)
        event = events[0]

        assert event["event_type"] == "query"
        assert event["data"]["query"] == "What is the meaning of life?"
        assert event["data"]["query_id"] == "q123"


def test_log_query_truncates_long_queries():
    """Test that long queries are truncated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLinesLogger(log_path)

        long_query = "a" * 500
        logger.log_query(long_query)

        events = read_log_file(log_path)
        event = events[0]

        assert len(event["data"]["query"]) == 200
        assert event["data"]["query_length"] == 500


def test_log_retrieval():
    """Test logging retrieval operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLinesLogger(log_path)

        logger.log_retrieval("test query", 5, "hybrid", 123.45)

        events = read_log_file(log_path)
        event = events[0]

        assert event["event_type"] == "retrieval"
        assert event["data"]["num_results"] == 5
        assert event["data"]["retrieval_method"] == "hybrid"
        assert event["data"]["duration_ms"] == 123.45


def test_log_model_call():
    """Test logging model API calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLinesLogger(log_path)

        logger.log_model_call("gpt-4", 100, 50, 1000.0)

        events = read_log_file(log_path)
        event = events[0]

        assert event["event_type"] == "model_call"
        assert event["data"]["model_name"] == "gpt-4"
        assert event["data"]["tokens_in"] == 100
        assert event["data"]["tokens_out"] == 50
        assert event["data"]["duration_ms"] == 1000.0
        assert event["data"]["tokens_per_second"] == 50.0


def test_get_logger_with_run_context():
    """Test getting logger with run context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"test": "config"}

        with run_context(config, run_type="test") as ctx:
            # Override logs directory for test
            logger = get_logger(logs_dir=Path(tmpdir))
            logger.log_event("test", {})

        # Check that log file was created with run_id
        log_path = Path(tmpdir) / f"{ctx.run_id}.jsonl"
        assert log_path.exists()


def test_log_event_convenience_function():
    """Test the log_event convenience function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"test": "config"}

        with run_context(config, run_type="test") as ctx:
            # Get logger first to set directory
            get_logger(logs_dir=Path(tmpdir))

            # Use convenience function
            log_event("convenience_test", {"key": "value"})

        log_path = Path(tmpdir) / f"{ctx.run_id}.jsonl"
        events = read_log_file(log_path)

        assert len(events) > 0
        event = events[0]
        assert event["event_type"] == "convenience_test"


def test_read_log_file_handles_missing_file():
    """Test that read_log_file handles missing files gracefully."""
    events = read_log_file(Path("/nonexistent/file.jsonl"))
    assert events == []
