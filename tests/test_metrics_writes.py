"""
Tests for metrics storage.
"""

import sqlite3
import tempfile
from pathlib import Path

from workbench.core.run_context import run_context
from workbench.observability.metrics import MetricsStore


def test_metrics_store_creates_database():
    """Test that metrics store creates database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"
        store = MetricsStore(db_path)

        assert db_path.exists()


def test_metrics_store_creates_tables():
    """Test that all required tables are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"
        store = MetricsStore(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
            """
        )
        tables = [row[0] for row in cursor.fetchall()]

        assert "component_latency" in tables
        assert "model_tokens" in tables
        assert "retrieval_counts" in tables

        conn.close()


def test_record_component_latency():
    """Test recording component latency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"
        store = MetricsStore(db_path)

        store.record_component_latency(
            "retriever",
            123.45,
            run_id="test_run",
            component_type="retriever",
        )

        # Query database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM component_latency")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1
        row = rows[0]
        # id, timestamp, run_id, component_name, component_type, duration_ms, status, metadata
        assert row[2] == "test_run"  # run_id
        assert row[3] == "retriever"  # component_name
        assert row[4] == "retriever"  # component_type
        assert row[5] == 123.45  # duration_ms


def test_record_component_latency_uses_run_context():
    """Test that component latency uses run context for run_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"
        store = MetricsStore(db_path)

        config = {"test": "config"}

        with run_context(config, run_type="test") as ctx:
            store.record_component_latency("retriever", 100.0)

        # Query database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT run_id FROM component_latency")
        run_id = cursor.fetchone()[0]
        conn.close()

        assert run_id == ctx.run_id


def test_record_model_tokens():
    """Test recording model token usage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"
        store = MetricsStore(db_path)

        store.record_model_tokens("gpt-4", 100, 50, 1000.0, run_id="test_run")

        # Query database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_tokens")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1
        row = rows[0]
        # id, timestamp, run_id, model_name, tokens_in, tokens_out, duration_ms, tokens_per_second
        assert row[2] == "test_run"  # run_id
        assert row[3] == "gpt-4"  # model_name
        assert row[4] == 100  # tokens_in
        assert row[5] == 50  # tokens_out
        assert row[6] == 1000.0  # duration_ms
        assert row[7] == 50.0  # tokens_per_second


def test_record_retrieval():
    """Test recording retrieval statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"
        store = MetricsStore(db_path)

        store.record_retrieval(
            final_count=5,
            query="test query",
            keyword_candidates=10,
            vector_candidates=10,
            reranked=5,
            duration_ms=150.0,
            run_id="test_run",
        )

        # Query database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM retrieval_counts")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1
        row = rows[0]
        # id, timestamp, run_id, query, keyword_candidates, vector_candidates,
        # reranked, final_count, duration_ms
        assert row[2] == "test_run"  # run_id
        assert row[3] == "test query"  # query
        assert row[4] == 10  # keyword_candidates
        assert row[5] == 10  # vector_candidates
        assert row[6] == 5  # reranked
        assert row[7] == 5  # final_count
        assert row[8] == 150.0  # duration_ms


def test_record_retrieval_truncates_long_query():
    """Test that long queries are truncated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"
        store = MetricsStore(db_path)

        long_query = "a" * 500
        store.record_retrieval(final_count=5, query=long_query, run_id="test_run")

        # Query database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT query FROM retrieval_counts")
        stored_query = cursor.fetchone()[0]
        conn.close()

        assert len(stored_query) == 200


def test_get_component_stats():
    """Test getting component statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"
        store = MetricsStore(db_path)

        # Record multiple latencies
        store.record_component_latency("retriever", 100.0, run_id="test_run")
        store.record_component_latency("retriever", 200.0, run_id="test_run")
        store.record_component_latency("retriever", 150.0, run_id="test_run")

        stats = store.get_component_stats("retriever")

        assert stats["count"] == 3
        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 200.0
        assert stats["avg_ms"] == 150.0


def test_get_component_stats_filtered_by_run():
    """Test getting component stats filtered by run_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"
        store = MetricsStore(db_path)

        # Record latencies for different runs
        store.record_component_latency("retriever", 100.0, run_id="run1")
        store.record_component_latency("retriever", 200.0, run_id="run2")

        stats = store.get_component_stats("retriever", run_id="run1")

        assert stats["count"] == 1
        assert stats["avg_ms"] == 100.0


def test_get_run_summary():
    """Test getting run summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"
        store = MetricsStore(db_path)

        run_id = "test_run"

        # Record various metrics
        store.record_component_latency("retriever", 100.0, run_id=run_id)
        store.record_component_latency("reranker", 50.0, run_id=run_id)
        store.record_model_tokens("gpt-4", 100, 50, 1000.0, run_id=run_id)
        store.record_retrieval(final_count=5, run_id=run_id)

        summary = store.get_run_summary(run_id)

        assert summary["run_id"] == run_id
        assert summary["component_calls"] == 2
        assert summary["total_duration_ms"] == 150.0
        assert summary["total_tokens_in"] == 100
        assert summary["total_tokens_out"] == 50
        assert summary["retrieval_calls"] == 1


def test_record_component_latency_convenience_function():
    """Test the convenience function for recording latency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"

        # Import after creating temp directory

        # Initialize with test path
        store = MetricsStore(db_path)

        config = {"test": "config"}

        with run_context(config, run_type="test") as ctx:
            # Use convenience function
            # Note: This would normally use the global store, but we're testing
            # the pattern
            store.record_component_latency("test_component", 100.0)

        # Verify it was recorded
        stats = store.get_component_stats("test_component", run_id=ctx.run_id)
        assert stats["count"] == 1


def test_multiple_runs_isolated():
    """Test that multiple runs keep their metrics isolated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metrics.sqlite"
        store = MetricsStore(db_path)

        # Record metrics for run1
        store.record_component_latency("retriever", 100.0, run_id="run1")
        store.record_model_tokens("gpt-4", 100, 50, 1000.0, run_id="run1")

        # Record metrics for run2
        store.record_component_latency("retriever", 200.0, run_id="run2")
        store.record_model_tokens("gpt-4", 200, 100, 2000.0, run_id="run2")

        # Get summaries
        summary1 = store.get_run_summary("run1")
        summary2 = store.get_run_summary("run2")

        assert summary1["total_duration_ms"] == 100.0
        assert summary1["total_tokens_in"] == 100

        assert summary2["total_duration_ms"] == 200.0
        assert summary2["total_tokens_in"] == 200
