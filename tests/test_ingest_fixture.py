"""
Tests for Wikipedia ingestion.

Tests the direct XML dump parser (primary path).
Uses small fixture files — no real dump needed.

Fixtures:
    tests/fixtures/simplewiki_sample.xml — tiny MediaWiki XML dump
"""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from workbench.data_build.ingest_wikipedia import (
    ingest_wikipedia_dump,
    iter_articles_from_dump,
    strip_wiki_markup,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
XML_FIXTURE = FIXTURES_DIR / "simplewiki_sample.xml"


# ═══════════════════════════════════════════════════════════════════════════
# strip_wiki_markup
# ═══════════════════════════════════════════════════════════════════════════


class TestStripWikiMarkup:
    """Tests for the wiki markup cleaning function."""

    def test_removes_bold_italic(self):
        assert "Anarchism" in strip_wiki_markup("'''Anarchism'''")
        assert "'''" not in strip_wiki_markup("'''bold'''")
        assert "''" not in strip_wiki_markup("''italic''")

    def test_converts_wikilinks(self):
        result = strip_wiki_markup("[[political philosophy]]")
        assert result == "political philosophy"

    def test_converts_piped_wikilinks(self):
        result = strip_wiki_markup("[[State (polity)|state]]")
        assert result == "state"

    def test_removes_categories(self):
        result = strip_wiki_markup("text\n[[Category:Science]]")
        assert "Category" not in result
        assert "text" in result

    def test_removes_file_links(self):
        result = strip_wiki_markup("[[File:Example.jpg|thumb|Caption]]")
        assert "File:" not in result

    def test_removes_templates(self):
        result = strip_wiki_markup("before {{Infobox|key=value}} after")
        assert "Infobox" not in result
        assert "before" in result
        assert "after" in result

    def test_removes_ref_tags(self):
        result = strip_wiki_markup("fact<ref>source</ref> more")
        assert "<ref>" not in result
        assert "source" not in result
        assert "fact" in result
        assert "more" in result

    def test_removes_html_tags(self):
        result = strip_wiki_markup("<div>content</div>")
        assert "content" in result
        assert "<div>" not in result

    def test_cleans_headings(self):
        result = strip_wiki_markup("== History ==")
        assert result == "History"

    def test_removes_external_links(self):
        result = strip_wiki_markup("[https://example.com click here]")
        assert result == "click here"
        assert "https" not in result

    def test_empty_string(self):
        assert strip_wiki_markup("") == ""

    def test_plain_text_unchanged(self):
        text = "This is plain text with no markup."
        assert strip_wiki_markup(text) == text

    def test_removes_html_comments(self):
        result = strip_wiki_markup("before <!-- hidden --> after")
        assert "hidden" not in result
        assert "before" in result


# ═══════════════════════════════════════════════════════════════════════════
# iter_articles_from_dump (XML parser)
# ═══════════════════════════════════════════════════════════════════════════


class TestIterArticlesFromDump:
    """Tests for the direct XML dump parser."""

    def test_yields_main_articles_only(self):
        """Should skip redirects (page 100) and talk pages (ns=1)."""
        articles = list(iter_articles_from_dump(XML_FIXTURE))
        titles = [a["title"] for a in articles]
        assert "Anarchism" in titles
        assert "Autism" in titles
        assert "Albedo" in titles
        assert "Stub article" in titles  # stub is still yielded; filtering is in ingest
        assert "Redirect page" not in titles
        assert "Talk:Anarchism" not in titles

    def test_article_count(self):
        articles = list(iter_articles_from_dump(XML_FIXTURE))
        assert len(articles) == 4

    def test_fields_present(self):
        for article in iter_articles_from_dump(XML_FIXTURE):
            assert "page_id" in article
            assert "title" in article
            assert "text" in article

    def test_markup_stripped(self):
        """Text should have wiki markup removed."""
        articles = list(iter_articles_from_dump(XML_FIXTURE))
        anarchism = [a for a in articles if a["title"] == "Anarchism"][0]
        assert "[[" not in anarchism["text"]
        assert "'''" not in anarchism["text"]
        assert "Category" not in anarchism["text"]

    def test_page_ids(self):
        articles = list(iter_articles_from_dump(XML_FIXTURE))
        ids = {a["page_id"] for a in articles}
        assert ids == {"12", "25", "39", "99"}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            list(iter_articles_from_dump(Path("/nonexistent/dump.xml")))

    def test_deterministic(self):
        first = list(iter_articles_from_dump(XML_FIXTURE))
        second = list(iter_articles_from_dump(XML_FIXTURE))
        assert first == second


# ═══════════════════════════════════════════════════════════════════════════
# ingest_wikipedia_dump (XML → parquet, end-to-end)
# ═══════════════════════════════════════════════════════════════════════════


class TestIngestWikipediaDump:
    """End-to-end tests for XML dump → parquet ingestion."""

    def test_creates_parquet(self, tmp_path):
        output = tmp_path / "articles.parquet"
        ingest_wikipedia_dump(XML_FIXTURE, output)
        assert output.exists()

    def test_filters_short_articles(self, tmp_path):
        output = tmp_path / "articles.parquet"
        summary = ingest_wikipedia_dump(XML_FIXTURE, output, min_text_length=50)
        table = pq.read_table(output)
        # "Stub article" has text="short" (5 chars) → filtered out
        assert table.num_rows == 3
        assert summary["num_articles"] == 3
        assert summary["skipped"] == 1

    def test_no_filter_keeps_all(self, tmp_path):
        output = tmp_path / "articles.parquet"
        summary = ingest_wikipedia_dump(XML_FIXTURE, output, min_text_length=0)
        assert summary["num_articles"] == 4
        assert summary["skipped"] == 0

    def test_parquet_columns(self, tmp_path):
        output = tmp_path / "articles.parquet"
        ingest_wikipedia_dump(XML_FIXTURE, output)
        table = pq.read_table(output)
        expected = {"page_id", "title", "text", "source", "ingested_at", "text_length"}
        assert set(table.column_names) == expected

    def test_source_label(self, tmp_path):
        output = tmp_path / "articles.parquet"
        ingest_wikipedia_dump(XML_FIXTURE, output, source_label="test-src")
        table = pq.read_table(output)
        assert all(s == "test-src" for s in table.column("source").to_pylist())

    def test_text_length_accurate(self, tmp_path):
        output = tmp_path / "articles.parquet"
        ingest_wikipedia_dump(XML_FIXTURE, output)
        table = pq.read_table(output)
        for text, length in zip(
            table.column("text").to_pylist(),
            table.column("text_length").to_pylist(),
        ):
            assert len(text) == length

    def test_summary_keys(self, tmp_path):
        output = tmp_path / "articles.parquet"
        summary = ingest_wikipedia_dump(XML_FIXTURE, output)
        for key in (
            "num_articles",
            "total_chars",
            "skipped",
            "output_path",
            "duration_s",
        ):
            assert key in summary
        assert summary["total_chars"] > 0

    def test_deterministic_content(self, tmp_path):
        out_a = tmp_path / "a.parquet"
        out_b = tmp_path / "b.parquet"
        ingest_wikipedia_dump(XML_FIXTURE, out_a)
        ingest_wikipedia_dump(XML_FIXTURE, out_b)
        table_a = pq.read_table(out_a)
        table_b = pq.read_table(out_b)
        for col in ("page_id", "title", "text", "source", "text_length"):
            assert table_a.column(col).to_pylist() == table_b.column(col).to_pylist()
