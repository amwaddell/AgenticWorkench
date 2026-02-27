"""
Wikipedia ingestion pipeline — direct XML dump parsing.

Reads a MediaWiki XML dump (e.g. simplewiki-latest-pages-articles.xml.bz2)
and produces a clean parquet dataset.  No WikiExtractor dependency required.

MediaWiki dump structure (simplified):
    <mediawiki>
      <page>
        <title>Anarchism</title>
        <ns>0</ns>                  ← namespace; 0 = main articles
        <id>12</id>
        <revision>
          <text>...wiki markup...</text>
        </revision>
      </page>
      ...
    </mediawiki>

Usage:
    from workbench.data_build.ingest_wikipedia import ingest_wikipedia_dump
    summary = ingest_wikipedia_dump(
        dump_path=Path("data/raw/wikipedia/simplewiki-latest-pages-articles.xml.bz2"),
        output_path=Path("data/processed/wikipedia_articles.parquet"),
    )
"""

import bz2
import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from xml.etree.ElementTree import iterparse

import pyarrow as pa
import pyarrow.parquet as pq

from workbench.observability.tracing import add_span_attributes, start_span

# ---------------------------------------------------------------------------
# MediaWiki namespace — tags in the dump are prefixed with this
# ---------------------------------------------------------------------------
MW_NS = "http://www.mediawiki.org/xml/export-0.11/"


def _tag(name: str) -> str:
    """Return the fully-qualified XML tag name."""
    return f"{{{MW_NS}}}{name}"


# Pre-compute the tags we care about
_PAGE = _tag("page")
_TITLE = _tag("title")
_NS = _tag("ns")
_ID = _tag("id")
_REDIRECT = _tag("redirect")
_REVISION = _tag("revision")
_TEXT = _tag("text")


# ---------------------------------------------------------------------------
# Wiki markup stripping
# ---------------------------------------------------------------------------


def strip_wiki_markup(text: str) -> str:
    """
    Remove common wiki markup to produce readable plain text.

    This is intentionally simple — it handles the most common patterns
    in Simple English Wikipedia.  For a production system you'd use
    mwparserfromhell, but this avoids an extra dependency.

    Args:
        text: Raw wikitext.

    Returns:
        Cleaned plain text.
    """
    if not text:
        return ""

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove <ref>...</ref> tags and self-closing <ref ... />
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/\s*>", "", text)

    # Remove remaining HTML tags (but keep their content)
    text = re.sub(r"<[^>]+>", "", text)

    # Remove {{...}} templates (handles simple nesting up to 2 levels)
    for _ in range(2):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # Remove tables {| ... |}
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

    # Remove categories and files: [[Category:...]] [[File:...]] [[Image:...]]
    text = re.sub(
        r"\[\[(Category|File|Image|file|image):[^\]]*\]\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Convert [[link|display]] → display, [[link]] → link
    text = re.sub(r"\[\[[^|\]]*\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Remove external links [http://... display] → display
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\]", "", text)

    # Bold/italic markers
    text = text.replace("'''", "").replace("''", "")

    # Section headings:  == Heading ==  → Heading
    text = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1", text, flags=re.MULTILINE)

    # Bullet / numbered list markers
    text = re.sub(r"^\*+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^:+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^;+\s*", "", text, flags=re.MULTILINE)

    # Magic words / behavior switches
    text = re.sub(r"__[A-Z]+__", "", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# XML dump iteration
# ---------------------------------------------------------------------------


def iter_articles_from_dump(dump_path: Path) -> Iterator[dict]:
    """
    Stream articles from a MediaWiki XML dump file.

    Only yields namespace-0 (main articles) and skips redirects.

    Supports both plain .xml and compressed .xml.bz2 files.

    Args:
        dump_path: Path to the dump file.

    Yields:
        Dict with keys: page_id, title, text (cleaned plain text).
    """
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump file not found: {dump_path}")

    # Open bz2-compressed or plain XML
    if dump_path.suffix == ".bz2":
        source = bz2.open(dump_path, "rt", encoding="utf-8")
    else:
        source = open(dump_path, encoding="utf-8")

    try:
        context = iterparse(source, events=("end",))

        page_id = None
        title = None
        ns = None
        is_redirect = False
        raw_text = None

        for event, elem in context:
            tag = elem.tag

            if tag == _TITLE:
                title = elem.text or ""

            elif tag == _NS:
                ns = elem.text or ""

            elif tag == _ID and page_id is None:
                page_id = elem.text or ""

            elif tag == _REDIRECT:
                is_redirect = True

            elif tag == _TEXT:
                raw_text = elem.text or ""

            elif tag == _PAGE:
                if ns == "0" and not is_redirect and title and raw_text:
                    cleaned = strip_wiki_markup(raw_text)
                    yield {
                        "page_id": page_id or "",
                        "title": title,
                        "text": cleaned,
                    }

                # Reset for next page
                page_id = None
                title = None
                ns = None
                is_redirect = False
                raw_text = None

                # Free memory — critical for large dumps
                elem.clear()

    finally:
        source.close()


# ---------------------------------------------------------------------------
# Arrow schema
# ---------------------------------------------------------------------------

ARTICLES_SCHEMA = pa.schema(
    [
        pa.field("page_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("text", pa.string()),
        pa.field("source", pa.string()),
        pa.field("ingested_at", pa.string()),
        pa.field("text_length", pa.int64()),
    ]
)


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------


def ingest_wikipedia_dump(
    dump_path: Path,
    output_path: Path,
    source_label: str = "simplewiki-latest",
    *,
    min_text_length: int = 50,
) -> dict:
    """
    Parse a MediaWiki XML dump and write a parquet articles dataset.

    Articles shorter than *min_text_length* characters (after markup
    stripping) are skipped — these are typically stubs.

    Args:
        dump_path: Path to the .xml.bz2 (or .xml) dump file.
        output_path: Where to write the parquet file.
        source_label: Value for the ``source`` column.
        min_text_length: Drop articles shorter than this (characters).

    Returns:
        Summary dict with keys: num_articles, total_chars, skipped,
        output_path, duration_s.
    """
    with start_span(
        "data.ingest_wikipedia",
        attributes={
            "dump_path": str(dump_path),
            "output_path": str(output_path),
            "source_label": source_label,
            "min_text_length": min_text_length,
        },
    ):
        t0 = time.time()
        ingested_at = datetime.now(UTC).isoformat()

        # Accumulate in columnar lists
        col_page_id: list[str] = []
        col_title: list[str] = []
        col_text: list[str] = []
        col_source: list[str] = []
        col_ingested: list[str] = []
        col_text_len: list[int] = []

        skipped = 0
        processed = 0

        for article in iter_articles_from_dump(dump_path):
            processed += 1
            text = article["text"]

            if len(text) < min_text_length:
                skipped += 1
                continue

            col_page_id.append(article["page_id"])
            col_title.append(article["title"])
            col_text.append(text)
            col_source.append(source_label)
            col_ingested.append(ingested_at)
            col_text_len.append(len(text))

            # Progress feedback every 25,000 articles
            if processed % 25_000 == 0:
                print(f"  processed {processed:,} pages, kept {len(col_page_id):,}...")

        # Build Arrow table and write parquet
        table = pa.table(
            {
                "page_id": pa.array(col_page_id, type=pa.string()),
                "title": pa.array(col_title, type=pa.string()),
                "text": pa.array(col_text, type=pa.string()),
                "source": pa.array(col_source, type=pa.string()),
                "ingested_at": pa.array(col_ingested, type=pa.string()),
                "text_length": pa.array(col_text_len, type=pa.int64()),
            },
            schema=ARTICLES_SCHEMA,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)

        duration_s = round(time.time() - t0, 2)
        num_articles = len(col_page_id)
        total_chars = sum(col_text_len)

        add_span_attributes(
            {
                "num_articles": num_articles,
                "total_chars": total_chars,
                "skipped": skipped,
                "duration_s": duration_s,
            }
        )

        return {
            "num_articles": num_articles,
            "total_chars": total_chars,
            "skipped": skipped,
            "output_path": str(output_path),
            "duration_s": duration_s,
        }
