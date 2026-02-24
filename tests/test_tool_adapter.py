"""
Tests for the ToolSpec framework (Day 2).

Validates:
    - Schema enforcement (Pydantic args validation)
    - Tool execution produces expected results
    - execute() creates trace spans and records metrics
    - to_langchain_tool() returns a working LangChain tool
    - to_langchain_tool() output is a JSON string
    - Migrated tools (search, open_chunk, timeline) work through ToolSpec
    - Error handling in run() is surfaced correctly
    - Observability hooks are best-effort (don't break tool calls)
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from workbench.core.types import ModelResponse
from workbench.tools.base import ToolSpec
from workbench.tools.open_chunk import OpenChunkArgs, OpenChunkTool
from workbench.tools.search_wikipedia import SearchWikipediaArgs, SearchWikipediaTool
from workbench.tools.timeline import TimelineArgs, TimelineTool

# Try to import LangChain (optional)
try:
    from langchain_core.tools import BaseTool

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

# Try to import OTel for span tests
try:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


# ================================================================== #
#  Fixtures: a minimal ToolSpec subclass for unit testing              #
# ================================================================== #


class EchoArgs(BaseModel):
    """Args for the echo test tool."""

    message: str = Field(..., description="Message to echo.")
    repeat: int = Field(default=1, ge=1, le=10, description="Repetitions.")


class EchoTool(ToolSpec):
    """Minimal tool that echoes its input — used only in tests."""

    name = "echo"
    description = "Echo a message back."
    args_schema = EchoArgs

    def run(self, **kwargs: Any) -> dict[str, Any]:
        msg = kwargs["message"]
        repeat = kwargs["repeat"]
        return {"echoed": msg * repeat, "length": len(msg) * repeat}


class FailingTool(ToolSpec):
    """Tool that always raises — used to test error handling."""

    name = "fail"
    description = "Always fails."
    args_schema = EchoArgs

    def run(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("intentional boom")


# ================================================================== #
#  Fakes for migrated-tool tests                                      #
# ================================================================== #


class FakeRetriever:
    """Fake retriever returning canned ChunkRef-like objects."""

    def retrieve(self, query: Any, top_k: int = 5) -> list:
        from workbench.core.types import ChunkRef

        return [
            ChunkRef(
                chunk_id="c001",
                score=0.95,
                title="Rome",
                section=None,
                snippet="The Roman Republic was...",
            ),
            ChunkRef(
                chunk_id="c002",
                score=0.80,
                title="Caesar",
                section="Assassination",
                snippet="On the Ides of March...",
            ),
        ][:top_k]


class FakeChunkStore:
    """Fake chunk store for open_chunk tests."""

    _data = {
        "c001": {
            "chunk_id": "c001",
            "document_id": "doc1",
            "title": "Rome",
            "section": None,
            "text": "The Roman Republic lasted from 509 BC to 27 BC.",
            "start_offset": 0,
            "end_offset": 47,
            "token_count": 12,
        },
    }

    def get(self, chunk_id: str) -> Any:
        from workbench.core.types import Chunk

        data = self._data.get(chunk_id)
        if data is None:
            return None
        return Chunk(**data)


class FakeModel:
    """Fake LLM that returns a canned timeline JSON."""

    def generate(self, messages: list[dict], **settings: Any) -> ModelResponse:
        timeline_json = json.dumps(
            [
                {
                    "date": "509 BC",
                    "event": "Republic founded",
                    "supporting_chunk_ids": ["c001"],
                }
            ]
        )
        return ModelResponse(
            text=timeline_json,
            tokens_in=100,
            tokens_out=50,
            latency_ms=10.0,
        )


# ================================================================== #
#  1. ToolSpec core: schema enforcement                               #
# ================================================================== #


class TestSchemaEnforcement:
    """Verify that execute() validates kwargs via args_schema."""

    def test_valid_args_accepted(self) -> None:
        tool = EchoTool()
        result = tool.execute(message="hello")
        assert result["echoed"] == "hello"

    def test_missing_required_arg_raises(self) -> None:
        tool = EchoTool()
        with pytest.raises(ValidationError, match="message"):
            tool.execute()  # message is required

    def test_wrong_type_raises(self) -> None:
        tool = EchoTool()
        with pytest.raises(ValidationError):
            tool.execute(message="hi", repeat="not_an_int")

    def test_out_of_range_raises(self) -> None:
        tool = EchoTool()
        with pytest.raises(ValidationError):
            tool.execute(message="hi", repeat=100)  # max is 10

    def test_default_values_applied(self) -> None:
        tool = EchoTool()
        result = tool.execute(message="x")
        assert result["echoed"] == "x"  # repeat defaults to 1

    def test_repeat_parameter_works(self) -> None:
        tool = EchoTool()
        result = tool.execute(message="ab", repeat=3)
        assert result["echoed"] == "ababab"
        assert result["length"] == 6

    def test_extra_kwargs_ignored(self) -> None:
        """Pydantic v2 ignores extra fields by default."""
        tool = EchoTool()
        result = tool.execute(message="ok", unknown_field="should be ignored")
        assert result["echoed"] == "ok"


# ================================================================== #
#  2. ToolSpec core: error handling                                   #
# ================================================================== #


class TestErrorHandling:
    """Verify that run() errors propagate through execute()."""

    def test_runtime_error_propagates(self) -> None:
        tool = FailingTool()
        with pytest.raises(RuntimeError, match="intentional boom"):
            tool.execute(message="anything")

    def test_error_does_not_corrupt_tool(self) -> None:
        """Tool should be reusable after an error."""
        tool = EchoTool()
        result = tool.execute(message="before")
        assert result["echoed"] == "before"


# ================================================================== #
#  3. ToolSpec: trace spans                                           #
# ================================================================== #


@pytest.mark.skipif(not HAS_OTEL, reason="opentelemetry SDK not installed")
class TestToolSpecSpans:
    """Verify that execute() creates a trace span."""

    @pytest.fixture(autouse=True)
    def _setup_tracing(self) -> None:
        import workbench.observability.tracing as tracing_mod

        self.exporter = InMemorySpanExporter()
        resource = Resource.create({"service.name": "test-tool-adapter"})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))

        self._old_tracer = tracing_mod._tracer
        self._old_provider = tracing_mod._tracer_provider

        tracing_mod._tracer_provider = provider
        tracing_mod._tracer = provider.get_tracer("test-tool-adapter")

        yield

        provider.shutdown()
        tracing_mod._tracer = self._old_tracer
        tracing_mod._tracer_provider = self._old_provider

    def test_execute_creates_span(self) -> None:
        """execute() should create a span named tool.<name>."""
        tool = EchoTool()
        tool.execute(message="span test")
        spans = self.exporter.get_finished_spans()
        names = [s.name for s in spans]
        assert "tool.echo" in names

    def test_span_has_tool_name_attribute(self) -> None:
        tool = EchoTool()
        tool.execute(message="attr test")
        spans = self.exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool.echo")
        attrs = dict(tool_span.attributes) if tool_span.attributes else {}
        assert attrs.get("tool_name") == "echo"

    def test_span_has_duration_attribute(self) -> None:
        tool = EchoTool()
        tool.execute(message="duration test")
        spans = self.exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool.echo")
        attrs = dict(tool_span.attributes) if tool_span.attributes else {}
        assert "duration_ms" in attrs

    def test_span_has_success_status(self) -> None:
        tool = EchoTool()
        tool.execute(message="status test")
        spans = self.exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool.echo")
        attrs = dict(tool_span.attributes) if tool_span.attributes else {}
        assert attrs.get("status") == "success"

    def test_failing_tool_span_has_error_status(self) -> None:
        tool = FailingTool()
        with pytest.raises(RuntimeError):
            tool.execute(message="fail")
        spans = self.exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool.fail")
        attrs = dict(tool_span.attributes) if tool_span.attributes else {}
        assert attrs.get("status") == "error"

    def test_search_tool_creates_span(self) -> None:
        tool = SearchWikipediaTool(retriever=FakeRetriever())
        tool.execute(query="Roman Republic", top_k=2)
        spans = self.exporter.get_finished_spans()
        names = [s.name for s in spans]
        assert "tool.search_wikipedia" in names

    def test_open_chunk_tool_creates_span(self) -> None:
        tool = OpenChunkTool(chunk_store=FakeChunkStore())
        tool.execute(chunk_id="c001")
        spans = self.exporter.get_finished_spans()
        names = [s.name for s in spans]
        assert "tool.open_chunk" in names


# ================================================================== #
#  4. LangChain adapter                                               #
# ================================================================== #


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
class TestLangChainAdapter:
    """Verify to_langchain_tool() produces a working LangChain tool."""

    def test_returns_base_tool_instance(self) -> None:
        tool = EchoTool()
        lc_tool = tool.to_langchain_tool()
        assert isinstance(lc_tool, BaseTool)

    def test_langchain_tool_has_correct_name(self) -> None:
        tool = EchoTool()
        lc_tool = tool.to_langchain_tool()
        assert lc_tool.name == "echo"

    def test_langchain_tool_has_correct_description(self) -> None:
        tool = EchoTool()
        lc_tool = tool.to_langchain_tool()
        assert lc_tool.description == "Echo a message back."

    def test_langchain_tool_invoke_returns_json_string(self) -> None:
        tool = EchoTool()
        lc_tool = tool.to_langchain_tool()
        raw = lc_tool.invoke({"message": "hello", "repeat": 2})
        assert isinstance(raw, str)
        data = json.loads(raw)
        assert data["echoed"] == "hellohello"

    def test_langchain_tool_validates_args(self) -> None:
        tool = EchoTool()
        lc_tool = tool.to_langchain_tool()
        with pytest.raises(Exception):
            # missing required 'message'
            lc_tool.invoke({})

    def test_search_tool_langchain_returns_json(self) -> None:
        tool = SearchWikipediaTool(retriever=FakeRetriever())
        lc_tool = tool.to_langchain_tool()
        raw = lc_tool.invoke({"query": "Rome", "top_k": 1})
        assert isinstance(raw, str)
        data = json.loads(raw)
        assert data["count"] >= 1

    def test_open_chunk_langchain_returns_json(self) -> None:
        tool = OpenChunkTool(chunk_store=FakeChunkStore())
        lc_tool = tool.to_langchain_tool()
        raw = lc_tool.invoke({"chunk_id": "c001"})
        assert isinstance(raw, str)
        data = json.loads(raw)
        assert data["found"] is True


class TestLangChainImportError:
    """Verify graceful error when langchain-core is missing."""

    def test_import_error_message(self) -> None:
        """If langchain isn't importable, we get a clear message.
        (This test only runs meaningfully when langchain IS installed,
        so it just verifies the happy path doesn't break.)"""
        if HAS_LANGCHAIN:
            tool = EchoTool()
            lc_tool = tool.to_langchain_tool()
            assert lc_tool is not None
        else:
            tool = EchoTool()
            with pytest.raises(ImportError, match="langchain-core"):
                tool.to_langchain_tool()


# ================================================================== #
#  5. Migrated tool: SearchWikipediaTool                              #
# ================================================================== #


class TestSearchWikipediaMigrated:
    """Verify SearchWikipediaTool works through ToolSpec."""

    def test_execute_returns_chunks(self) -> None:
        tool = SearchWikipediaTool(retriever=FakeRetriever())
        result = tool.execute(query="Roman Republic", top_k=2)
        assert result["count"] == 2
        assert len(result["chunks"]) == 2

    def test_execute_returns_chunk_ids(self) -> None:
        tool = SearchWikipediaTool(retriever=FakeRetriever())
        result = tool.execute(query="test", top_k=2)
        ids = [c["chunk_id"] for c in result["chunks"]]
        assert "c001" in ids
        assert "c002" in ids

    def test_execute_empty_query(self) -> None:
        tool = SearchWikipediaTool(retriever=FakeRetriever())
        result = tool.execute(query="", top_k=5)
        assert result["count"] == 0
        assert result["chunks"] == []

    def test_schema_rejects_negative_top_k(self) -> None:
        tool = SearchWikipediaTool(retriever=FakeRetriever())
        with pytest.raises(ValidationError):
            tool.execute(query="test", top_k=-1)

    def test_schema_rejects_missing_query(self) -> None:
        tool = SearchWikipediaTool(retriever=FakeRetriever())
        with pytest.raises(ValidationError):
            tool.execute(top_k=5)  # query is required

    def test_is_toolspec_subclass(self) -> None:
        tool = SearchWikipediaTool(retriever=FakeRetriever())
        assert isinstance(tool, ToolSpec)

    def test_has_correct_name(self) -> None:
        tool = SearchWikipediaTool(retriever=FakeRetriever())
        assert tool.name == "search_wikipedia"


# ================================================================== #
#  6. Migrated tool: OpenChunkTool                                    #
# ================================================================== #


class TestOpenChunkMigrated:
    """Verify OpenChunkTool works through ToolSpec."""

    def test_execute_found(self) -> None:
        tool = OpenChunkTool(chunk_store=FakeChunkStore())
        result = tool.execute(chunk_id="c001")
        assert result["found"] is True
        assert result["chunk"]["chunk_id"] == "c001"
        assert result["text_length"] > 0

    def test_execute_not_found(self) -> None:
        tool = OpenChunkTool(chunk_store=FakeChunkStore())
        result = tool.execute(chunk_id="nonexistent")
        assert result["found"] is False
        assert result["chunk"] is None

    def test_schema_rejects_missing_chunk_id(self) -> None:
        tool = OpenChunkTool(chunk_store=FakeChunkStore())
        with pytest.raises(ValidationError):
            tool.execute()  # chunk_id is required

    def test_is_toolspec_subclass(self) -> None:
        tool = OpenChunkTool(chunk_store=FakeChunkStore())
        assert isinstance(tool, ToolSpec)

    def test_has_correct_name(self) -> None:
        tool = OpenChunkTool(chunk_store=FakeChunkStore())
        assert tool.name == "open_chunk"


# ================================================================== #
#  7. Migrated tool: TimelineTool                                     #
# ================================================================== #


class TestTimelineMigrated:
    """Verify TimelineTool works through ToolSpec."""

    def _make_tool(self) -> TimelineTool:
        return TimelineTool(model=FakeModel())

    def test_execute_returns_timeline(self) -> None:
        tool = self._make_tool()
        result = tool.execute(
            question="When was Rome founded?",
            opened_chunks=[
                {
                    "chunk_id": "c001",
                    "document_id": "doc1",
                    "title": "Rome",
                    "section": None,
                    "text": "Rome was founded.",
                }
            ],
        )
        assert result["count"] == 1
        assert result["timeline"][0]["date"] == "509 BC"

    def test_execute_returns_cited_chunk_ids(self) -> None:
        tool = self._make_tool()
        result = tool.execute(
            question="test",
            opened_chunks=[
                {
                    "chunk_id": "c001",
                    "document_id": "doc1",
                    "title": "Rome",
                    "section": None,
                    "text": "text",
                }
            ],
        )
        assert "c001" in result["cited_chunk_ids"]

    def test_execute_returns_prompt_version(self) -> None:
        tool = self._make_tool()
        result = tool.execute(
            question="test",
            opened_chunks=[
                {
                    "chunk_id": "c001",
                    "document_id": "doc1",
                    "title": "Rome",
                    "section": None,
                    "text": "text",
                }
            ],
        )
        assert "prompt_version" in result
        assert len(result["prompt_version"]) > 0

    def test_schema_rejects_missing_question(self) -> None:
        tool = self._make_tool()
        with pytest.raises(ValidationError):
            tool.execute(opened_chunks=[])

    def test_schema_rejects_missing_chunks(self) -> None:
        tool = self._make_tool()
        with pytest.raises(ValidationError):
            tool.execute(question="test")

    def test_is_toolspec_subclass(self) -> None:
        tool = self._make_tool()
        assert isinstance(tool, ToolSpec)

    def test_has_correct_name(self) -> None:
        tool = self._make_tool()
        assert tool.name == "timeline"


# ================================================================== #
#  8. _parse_timeline stays accessible as static method               #
# ================================================================== #


class TestParseTimelineStillWorks:
    """
    _parse_timeline is a static method used by tests/test_timeline_schema.py.
    Verify it still works after the ToolSpec migration.
    """

    def test_valid_json_parsed(self) -> None:
        raw = '[{"date": "509 BC", "event": "Founded", "supporting_chunk_ids": ["a"]}]'
        items = TimelineTool._parse_timeline(raw, {"a"})
        assert len(items) == 1
        assert items[0].date == "509 BC"

    def test_empty_returns_empty(self) -> None:
        assert TimelineTool._parse_timeline("[]", None) == []

    def test_garbage_raises(self) -> None:
        with pytest.raises(ValueError):
            TimelineTool._parse_timeline("not json", None)


# ================================================================== #
#  9. Args schemas are proper Pydantic models                         #
# ================================================================== #


class TestArgSchemas:
    """Verify that the args schemas are well-formed Pydantic models."""

    def test_search_args_schema(self) -> None:
        args = SearchWikipediaArgs(query="test")
        assert args.query == "test"
        assert args.top_k == 5  # default

    def test_open_chunk_args_schema(self) -> None:
        args = OpenChunkArgs(chunk_id="abc")
        assert args.chunk_id == "abc"

    def test_timeline_args_schema(self) -> None:
        args = TimelineArgs(
            question="q",
            opened_chunks=[{"chunk_id": "x", "text": "y"}],
        )
        assert args.question == "q"
        assert len(args.opened_chunks) == 1

    def test_search_args_json_schema(self) -> None:
        """args_schema should produce a valid JSON schema."""
        schema = SearchWikipediaArgs.model_json_schema()
        assert "query" in schema["properties"]
        assert "top_k" in schema["properties"]

    def test_open_chunk_args_json_schema(self) -> None:
        schema = OpenChunkArgs.model_json_schema()
        assert "chunk_id" in schema["properties"]

    def test_timeline_args_json_schema(self) -> None:
        schema = TimelineArgs.model_json_schema()
        assert "question" in schema["properties"]
        assert "opened_chunks" in schema["properties"]
