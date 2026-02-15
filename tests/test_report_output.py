"""
Tests for evaluation report generation.

Verifies that:
    - a single-question report contains expected sections
    - a summary report contains the results table
    - the report can be saved to disk and read back
"""

from pathlib import Path

from workbench.agents.state import (
    AgentState,
    OpenedChunk,
    RetrievedChunkSummary,
    TimelineItem,
)
from workbench.evaluation.measures import compute_all_measures
from workbench.evaluation.reports import (
    generate_report,
    generate_summary_report,
    save_report,
)


def _make_test_state() -> AgentState:
    """Create a realistic AgentState for testing."""
    state = AgentState(question="When did the Roman Republic end?")
    state.step = 4
    state.done = True

    state.retrieved = [
        RetrievedChunkSummary(
            chunk_id="chunk_001",
            score=0.9,
            title="Roman Republic",
            snippet="The Roman Republic...",
        ),
        RetrievedChunkSummary(
            chunk_id="chunk_002", score=0.8, title="Augustus", snippet="Augustus was..."
        ),
    ]

    state.opened = [
        OpenedChunk(
            chunk_id="chunk_001",
            document_id="doc_1",
            title="Roman Republic",
            section="Fall",
            text="The Roman Republic ended in 27 BC when Octavian became Augustus.",
        ),
        OpenedChunk(
            chunk_id="chunk_002",
            document_id="doc_2",
            title="Augustus",
            section="Rise to Power",
            text="Octavian defeated Mark Antony at Actium in 31 BC.",
        ),
    ]

    state.timeline = [
        TimelineItem(
            date="44 BC",
            event="Julius Caesar assassinated",
            supporting_chunk_ids=["chunk_001"],
        ),
        TimelineItem(
            date="31 BC",
            event="Battle of Actium",
            supporting_chunk_ids=["chunk_002"],
        ),
        TimelineItem(
            date="27 BC",
            event="Octavian becomes Augustus",
            supporting_chunk_ids=["chunk_001", "chunk_002"],
        ),
    ]

    state.answer_text = (
        "The Roman Republic ended in 27 BC [chunk_001]. "
        "Octavian defeated Mark Antony at Actium in 31 BC [chunk_002] "
        "and became Augustus [chunk_001][chunk_002]."
    )
    state.citations = ["chunk_001", "chunk_002"]
    state.tokens_in = 500
    state.tokens_out = 120
    state.model_latency_ms = 3500.0

    return state


class TestGenerateReport:
    """Tests for single-question report generation."""

    def test_report_contains_question_header(self):
        state = _make_test_state()
        metrics = compute_all_measures(state)
        md = generate_report(state, metrics, run_id="test_run_001")
        assert "## Question" in md
        assert "Roman Republic" in md

    def test_report_contains_answer_section(self):
        state = _make_test_state()
        metrics = compute_all_measures(state)
        md = generate_report(state, metrics, run_id="test_run_001")
        assert "## Answer" in md
        assert "27 BC" in md

    def test_report_contains_timeline_table(self):
        state = _make_test_state()
        metrics = compute_all_measures(state)
        md = generate_report(state, metrics, run_id="test_run_001")
        assert "## Timeline" in md
        assert "44 BC" in md
        assert "Battle of Actium" in md

    def test_report_contains_citations_section(self):
        state = _make_test_state()
        metrics = compute_all_measures(state)
        md = generate_report(state, metrics, run_id="test_run_001")
        assert "## Citations" in md
        assert "chunk_001" in md
        assert "chunk_002" in md

    def test_report_contains_sources_section(self):
        state = _make_test_state()
        metrics = compute_all_measures(state)
        md = generate_report(state, metrics, run_id="test_run_001")
        assert "## Sources Used" in md
        assert "Roman Republic" in md
        assert "Augustus" in md

    def test_report_contains_metrics_table(self):
        state = _make_test_state()
        metrics = compute_all_measures(state)
        md = generate_report(state, metrics, run_id="test_run_001")
        assert "## Metrics" in md
        assert "Citations in answer" in md

    def test_report_with_question_meta(self):
        state = _make_test_state()
        metrics = compute_all_measures(state)
        md = generate_report(
            state,
            metrics,
            run_id="test_run_001",
            question_meta={
                "topic": "Ancient Rome",
                "expected_date_range": "133 BC – 27 BC",
            },
        )
        assert "Ancient Rome" in md
        assert "133 BC" in md

    def test_report_with_no_timeline(self):
        state = _make_test_state()
        state.timeline = []
        metrics = compute_all_measures(state)
        md = generate_report(state, metrics, run_id="test_run_001")
        assert "No timeline extracted" in md

    def test_report_with_error(self):
        state = _make_test_state()
        state.error = "max_steps reached (10)"
        metrics = compute_all_measures(state)
        md = generate_report(state, metrics, run_id="test_run_001")
        assert "## Error" in md
        assert "max_steps" in md


class TestGenerateSummaryReport:
    """Tests for multi-question summary report."""

    def test_summary_contains_header(self):
        results = [
            {
                "question": "Q1?",
                "metrics": {
                    "citation_count": 2,
                    "unique_cited_titles": 1,
                    "timeline_item_count": 3,
                    "total_steps": 4,
                },
                "error": None,
            },
            {
                "question": "Q2?",
                "metrics": {
                    "citation_count": 1,
                    "unique_cited_titles": 1,
                    "timeline_item_count": 2,
                    "total_steps": 4,
                },
                "error": None,
            },
        ]
        md = generate_summary_report(results, run_id="test_summary")
        assert "# Evaluation Summary" in md
        assert "Questions evaluated: 2" in md

    def test_summary_contains_results_table(self):
        results = [
            {
                "question": "Q1?",
                "metrics": {
                    "citation_count": 2,
                    "unique_cited_titles": 1,
                    "timeline_item_count": 3,
                    "total_steps": 4,
                },
                "error": None,
            },
        ]
        md = generate_summary_report(results, run_id="test_summary")
        assert "| #" in md
        assert "Q1?" in md

    def test_summary_contains_averages(self):
        results = [
            {
                "question": "Q1?",
                "metrics": {
                    "citation_count": 4,
                    "unique_cited_titles": 2,
                    "timeline_item_count": 3,
                    "total_steps": 4,
                },
                "error": None,
            },
            {
                "question": "Q2?",
                "metrics": {
                    "citation_count": 2,
                    "unique_cited_titles": 1,
                    "timeline_item_count": 1,
                    "total_steps": 4,
                },
                "error": None,
            },
        ]
        md = generate_summary_report(results, run_id="test_summary")
        assert "## Averages" in md
        assert "3.0" in md  # avg citations = (4+2)/2


class TestSaveReport:
    """Tests for saving reports to disk."""

    def test_save_and_read_back(self, tmp_path: Path):
        state = _make_test_state()
        metrics = compute_all_measures(state)
        md = generate_report(state, metrics, run_id="test_save")

        out_path = tmp_path / "reports" / "test.md"
        returned = save_report(md, out_path)

        assert returned == out_path
        assert out_path.exists()

        content = out_path.read_text(encoding="utf-8")
        assert "## Question" in content
        assert "Roman Republic" in content
