"""
Tests for timeline tool output parsing.

Verifies that:
    - valid JSON is parsed into TimelineItem objects
    - malformed output is rejected with ValueError
    - markdown code fences are stripped
    - trailing commas are tolerated
    - unknown chunk_ids are filtered out
    - empty timelines work
"""

import pytest

from workbench.agents.state import TimelineItem
from workbench.tools.timeline import TimelineTool


class TestTimelineParsing:
    """Unit tests for TimelineTool._parse_timeline."""

    KNOWN_IDS = {"chunk_a", "chunk_b", "chunk_c"}

    def test_valid_json_parsed(self):
        """Well-formed JSON array produces TimelineItem list."""
        raw = """
        [
            {
                "date": "509 BC",
                "event": "Republic founded",
                "supporting_chunk_ids": ["chunk_a"]
            },
            {
                "date": "27 BC",
                "event": "Augustus takes power",
                "supporting_chunk_ids": ["chunk_b", "chunk_c"]
            }
        ]
        """
        items = TimelineTool._parse_timeline(raw, self.KNOWN_IDS)
        assert len(items) == 2
        assert items[0].date == "509 BC"
        assert items[0].event == "Republic founded"
        assert items[0].supporting_chunk_ids == ["chunk_a"]
        assert items[1].supporting_chunk_ids == ["chunk_b", "chunk_c"]

    def test_all_items_are_timeline_items(self):
        """Each parsed item is a proper TimelineItem instance."""
        raw = '[{"date": "1066", "event": "Battle of Hastings", "supporting_chunk_ids": ["chunk_a"]}]'
        items = TimelineTool._parse_timeline(raw, self.KNOWN_IDS)
        assert all(isinstance(item, TimelineItem) for item in items)

    def test_markdown_code_fences_stripped(self):
        """JSON wrapped in ```json ... ``` fences is still parsed."""
        raw = '```json\n[{"date": "44 BC", "event": "Caesar assassinated", "supporting_chunk_ids": ["chunk_a"]}]\n```'
        items = TimelineTool._parse_timeline(raw, self.KNOWN_IDS)
        assert len(items) == 1
        assert items[0].date == "44 BC"

    def test_trailing_comma_tolerated(self):
        """A trailing comma after the last entry is handled."""
        raw = """[
            {"date": "100 BC", "event": "Event A", "supporting_chunk_ids": ["chunk_a"]},
        ]"""
        items = TimelineTool._parse_timeline(raw, self.KNOWN_IDS)
        assert len(items) == 1

    def test_empty_array_returns_empty_list(self):
        """An empty JSON array produces an empty list."""
        items = TimelineTool._parse_timeline("[]", self.KNOWN_IDS)
        assert items == []

    def test_empty_string_returns_empty_list(self):
        """An empty string produces an empty list."""
        items = TimelineTool._parse_timeline("", self.KNOWN_IDS)
        assert items == []

    def test_unknown_chunk_ids_filtered(self):
        """Chunk IDs not in known_chunk_ids are removed."""
        raw = '[{"date": "100 AD", "event": "Something", "supporting_chunk_ids": ["chunk_a", "unknown_xyz"]}]'
        items = TimelineTool._parse_timeline(raw, self.KNOWN_IDS)
        assert items[0].supporting_chunk_ids == ["chunk_a"]

    def test_missing_date_skipped(self):
        """Entries without a 'date' key are skipped."""
        raw = '[{"event": "No date here", "supporting_chunk_ids": []}]'
        items = TimelineTool._parse_timeline(raw, self.KNOWN_IDS)
        assert len(items) == 0

    def test_missing_event_skipped(self):
        """Entries without an 'event' key are skipped."""
        raw = '[{"date": "100 AD", "supporting_chunk_ids": []}]'
        items = TimelineTool._parse_timeline(raw, self.KNOWN_IDS)
        assert len(items) == 0

    def test_non_list_raises_value_error(self):
        """A JSON object (not array) raises ValueError."""
        raw = '{"date": "100 AD", "event": "Single object"}'
        with pytest.raises(ValueError, match="Expected JSON array"):
            TimelineTool._parse_timeline(raw, self.KNOWN_IDS)

    def test_total_garbage_raises_value_error(self):
        """Completely unparseable text raises ValueError."""
        raw = "This is not JSON at all, just random text."
        with pytest.raises(ValueError, match="Could not parse timeline JSON"):
            TimelineTool._parse_timeline(raw, self.KNOWN_IDS)

    def test_no_known_ids_filter_keeps_all(self):
        """When known_chunk_ids is None, all IDs are kept."""
        raw = '[{"date": "100 AD", "event": "Foo", "supporting_chunk_ids": ["any_id_at_all"]}]'
        items = TimelineTool._parse_timeline(raw, known_chunk_ids=None)
        assert items[0].supporting_chunk_ids == ["any_id_at_all"]

    def test_non_dict_entries_skipped(self):
        """Non-dict entries in the array are silently skipped."""
        raw = '[{"date": "100 AD", "event": "Good"}, "bad_entry", 42]'
        items = TimelineTool._parse_timeline(raw, known_chunk_ids=None)
        assert len(items) == 1

    def test_missing_chunk_ids_defaults_to_empty(self):
        """Entries without supporting_chunk_ids still parse."""
        raw = '[{"date": "100 AD", "event": "No chunks cited"}]'
        items = TimelineTool._parse_timeline(raw, known_chunk_ids=None)
        assert len(items) == 1
        assert items[0].supporting_chunk_ids == []
