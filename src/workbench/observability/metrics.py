"""
Metrics storage using SQLite.

Stores performance metrics, token counts, and retrieval statistics
for analysis and optimization.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from workbench.core.run_context import get_run_context


class MetricsStore:
    """
    SQLite-based metrics storage.

    Stores component latencies, token counts, and retrieval statistics.
    """

    def __init__(self, db_path: Path):
        """
        Initialize metrics store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Component latency table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS component_latency (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                run_id TEXT NOT NULL,
                component_name TEXT NOT NULL,
                component_type TEXT,
                duration_ms REAL NOT NULL,
                status TEXT DEFAULT 'success',
                metadata TEXT
            )
        """)

        # Model token usage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                tokens_in INTEGER NOT NULL,
                tokens_out INTEGER NOT NULL,
                duration_ms REAL,
                tokens_per_second REAL
            )
        """)

        # Retrieval counts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                run_id TEXT NOT NULL,
                query TEXT,
                keyword_candidates INTEGER DEFAULT 0,
                vector_candidates INTEGER DEFAULT 0,
                reranked INTEGER DEFAULT 0,
                final_count INTEGER NOT NULL,
                duration_ms REAL
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_latency_run_id 
            ON component_latency(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_latency_component 
            ON component_latency(component_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tokens_run_id 
            ON model_tokens(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_retrieval_run_id 
            ON retrieval_counts(run_id)
        """)

        conn.commit()
        conn.close()

    def record_component_latency(
        self,
        component_name: str,
        duration_ms: float,
        run_id: str | None = None,
        component_type: str | None = None,
        status: str = "success",
        metadata: str | None = None,
    ) -> None:
        """
        Record component execution latency.

        Args:
            component_name: Name of the component
            duration_ms: Duration in milliseconds
            run_id: Run ID (uses context if not provided)
            component_type: Type of component
            status: Execution status (success, error)
            metadata: Optional JSON metadata
        """
        if run_id is None:
            run_ctx = get_run_context()
            if run_ctx is not None:
                run_id = run_ctx.run_id
            else:
                run_id = "unknown"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO component_latency 
            (timestamp, run_id, component_name, component_type, duration_ms, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                run_id,
                component_name,
                component_type,
                duration_ms,
                status,
                metadata,
            ),
        )

        conn.commit()
        conn.close()

    def record_model_tokens(
        self,
        model_name: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: float,
        run_id: str | None = None,
    ) -> None:
        """
        Record model token usage.

        Args:
            model_name: Name of the model
            tokens_in: Input token count
            tokens_out: Output token count
            duration_ms: Duration in milliseconds
            run_id: Run ID (uses context if not provided)
        """
        if run_id is None:
            run_ctx = get_run_context()
            if run_ctx is not None:
                run_id = run_ctx.run_id
            else:
                run_id = "unknown"

        tokens_per_second = tokens_out / (duration_ms / 1000) if duration_ms > 0 else 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO model_tokens 
            (timestamp, run_id, model_name, tokens_in, tokens_out, duration_ms, tokens_per_second)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                run_id,
                model_name,
                tokens_in,
                tokens_out,
                duration_ms,
                tokens_per_second,
            ),
        )

        conn.commit()
        conn.close()

    def record_retrieval(
        self,
        final_count: int,
        query: str | None = None,
        keyword_candidates: int = 0,
        vector_candidates: int = 0,
        reranked: int = 0,
        duration_ms: float | None = None,
        run_id: str | None = None,
    ) -> None:
        """
        Record retrieval statistics.

        Args:
            final_count: Final number of chunks returned
            query: Query text (truncated for storage)
            keyword_candidates: Number of keyword search results
            vector_candidates: Number of vector search results
            reranked: Number of results after reranking
            duration_ms: Duration in milliseconds
            run_id: Run ID (uses context if not provided)
        """
        if run_id is None:
            run_ctx = get_run_context()
            if run_ctx is not None:
                run_id = run_ctx.run_id
            else:
                run_id = "unknown"

        # Truncate query for storage
        if query and len(query) > 200:
            query = query[:200]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO retrieval_counts 
            (timestamp, run_id, query, keyword_candidates, vector_candidates, 
             reranked, final_count, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                run_id,
                query,
                keyword_candidates,
                vector_candidates,
                reranked,
                final_count,
                duration_ms,
            ),
        )

        conn.commit()
        conn.close()

    def get_component_stats(
        self, component_name: str, run_id: str | None = None
    ) -> dict:
        """
        Get statistics for a component.

        Args:
            component_name: Component name
            run_id: Optional run ID to filter by

        Returns:
            Dictionary with min, max, avg, count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT 
                COUNT(*) as count,
                MIN(duration_ms) as min_ms,
                MAX(duration_ms) as max_ms,
                AVG(duration_ms) as avg_ms
            FROM component_latency
            WHERE component_name = ?
        """
        params = [component_name]

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)

        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()

        return {
            "count": row[0] or 0,
            "min_ms": row[1] or 0,
            "max_ms": row[2] or 0,
            "avg_ms": row[3] or 0,
        }

    def get_run_summary(self, run_id: str) -> dict:
        """
        Get summary statistics for a run.

        Args:
            run_id: Run ID

        Returns:
            Dictionary with run statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get component count and total time
        cursor.execute(
            """
            SELECT COUNT(*), SUM(duration_ms)
            FROM component_latency
            WHERE run_id = ?
            """,
            (run_id,),
        )
        comp_row = cursor.fetchone()

        # Get token usage
        cursor.execute(
            """
            SELECT SUM(tokens_in), SUM(tokens_out)
            FROM model_tokens
            WHERE run_id = ?
            """,
            (run_id,),
        )
        token_row = cursor.fetchone()

        # Get retrieval count
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM retrieval_counts
            WHERE run_id = ?
            """,
            (run_id,),
        )
        retrieval_row = cursor.fetchone()

        conn.close()

        return {
            "run_id": run_id,
            "component_calls": comp_row[0] or 0,
            "total_duration_ms": comp_row[1] or 0,
            "total_tokens_in": token_row[0] or 0,
            "total_tokens_out": token_row[1] or 0,
            "retrieval_calls": retrieval_row[0] or 0,
        }


# Global metrics store instance
_metrics_store: MetricsStore | None = None


def get_metrics_store(db_path: Path | None = None) -> MetricsStore:
    """
    Get or create the global metrics store.

    Args:
        db_path: Optional database path (defaults to runs/metrics/metrics.sqlite)

    Returns:
        MetricsStore instance
    """
    global _metrics_store

    if _metrics_store is None:
        if db_path is None:
            db_path = Path("runs/metrics/metrics.sqlite")
        _metrics_store = MetricsStore(db_path)

    return _metrics_store


def record_component_latency(
    component_name: str,
    duration_ms: float,
    component_type: str | None = None,
) -> None:
    """
    Convenience function to record component latency.

    Args:
        component_name: Component name
        duration_ms: Duration in milliseconds
        component_type: Optional component type
    """
    store = get_metrics_store()
    store.record_component_latency(
        component_name, duration_ms, component_type=component_type
    )
