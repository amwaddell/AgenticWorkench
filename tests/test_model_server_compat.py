"""
Model server compatibility tests (Day 9).

Three test layers:
    A. Adapter unit tests — payload format, response parsing, error mapping
    B. Server integration tests — real server round-trips (skipped if offline)
    C. Infinity endpoint tests — embedding + reranker HTTP contract

These tests verify the **contract** your app depends on, not
implementation details. If you upgrade servers, models, or endpoints,
these tests catch regressions.

Run unit tests (always):
    pytest tests/test_model_server_compat.py -m "not integration"

Run all including integration (requires running servers):
    pytest tests/test_model_server_compat.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

# Canonical OpenAI-compatible response shape
VALID_CHAT_RESPONSE = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "local-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Rome fell in 476 AD."},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 42,
        "completion_tokens": 15,
        "total_tokens": 57,
    },
}

# Response with missing usage (some servers omit this)
RESPONSE_NO_USAGE = {
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Some answer."},
            "finish_reason": "stop",
        }
    ],
}

# Response with empty choices
RESPONSE_EMPTY_CHOICES = {
    "choices": [],
    "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
}

# Multi-turn messages (the shape the adapter must accept)
MULTI_TURN_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about WW2."},
    {"role": "assistant", "content": "World War II was a global conflict."},
    {"role": "user", "content": "What was the Battle of Stalingrad?"},
]


# ================================================================== #
#  A. ADAPTER UNIT TESTS                                               #
# ================================================================== #


class TestRequestPayloadFormat:
    """Verify _create_request builds correct payloads."""

    def _make_model(self):
        from workbench.models.llamacpp_server import LlamaCppServerModel

        return LlamaCppServerModel(
            base_url="http://127.0.0.1:8080",
            model_name="test-model",
            timeout_seconds=10,
        )

    def test_basic_request_shape(self):
        model = self._make_model()
        from workbench.models.language_model import GenerationSettings

        settings = GenerationSettings(temperature=0.5, max_tokens=100)
        payload = model._create_request(
            [{"role": "user", "content": "hello"}], settings
        )

        assert payload["model"] == "test-model"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100
        assert payload["stream"] is False

    def test_multi_turn_messages_accepted(self):
        """Adapter should accept multi-turn message arrays."""
        model = self._make_model()
        from workbench.models.language_model import GenerationSettings

        settings = GenerationSettings(temperature=0.7, max_tokens=200)
        payload = model._create_request(MULTI_TURN_MESSAGES, settings)

        assert len(payload["messages"]) == 4
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][-1]["role"] == "user"

    def test_optional_params_omitted_when_none(self):
        """top_p, top_k, stop should not appear when None."""
        model = self._make_model()
        from workbench.models.language_model import GenerationSettings

        settings = GenerationSettings(temperature=0.7, max_tokens=100)
        payload = model._create_request([{"role": "user", "content": "test"}], settings)
        assert "top_p" not in payload
        assert "top_k" not in payload
        assert "stop" not in payload


class TestResponseParsing:
    """Verify _parse_response handles various server responses."""

    def _make_model(self):
        from workbench.models.llamacpp_server import LlamaCppServerModel

        return LlamaCppServerModel(base_url="http://127.0.0.1:8080")

    def test_valid_response_parsed(self):
        model = self._make_model()
        result = model._parse_response(VALID_CHAT_RESPONSE, latency_ms=50.0)

        assert result.text == "Rome fell in 476 AD."
        assert result.tokens_in == 42
        assert result.tokens_out == 15
        assert result.latency_ms == 50.0
        assert result.finish_reason == "stop"

    def test_missing_usage_handled(self):
        """Adapter should not crash if usage is missing."""
        model = self._make_model()
        # This relies on count_tokens_from_usage handling empty usage
        result = model._parse_response(RESPONSE_NO_USAGE, latency_ms=30.0)
        assert result.text == "Some answer."
        # Token counts should be estimated or zero, not an exception
        assert isinstance(result.tokens_in, int)
        assert isinstance(result.tokens_out, int)

    def test_empty_choices_raises(self):
        """Empty choices should raise ModelResponseError."""
        from workbench.models.errors import ModelResponseError

        model = self._make_model()
        with pytest.raises(ModelResponseError, match="No choices"):
            model._parse_response(RESPONSE_EMPTY_CHOICES, latency_ms=10.0)

    def test_response_metadata_populated(self):
        model = self._make_model()
        result = model._parse_response(VALID_CHAT_RESPONSE, latency_ms=50.0)
        assert result.metadata["model"] == "local-model"
        assert result.metadata["provider"] == "llamacpp_server"


class TestErrorMapping:
    """Verify error types are mapped correctly."""

    def test_timeout_maps_to_model_timeout_error(self):
        import httpx

        from workbench.models.errors import ModelTimeoutError
        from workbench.models.llamacpp_server import LlamaCppServerModel

        model = LlamaCppServerModel(base_url="http://127.0.0.1:8080")
        model.client = MagicMock()
        model.client.post.side_effect = httpx.TimeoutException("timed out")

        with pytest.raises(ModelTimeoutError):
            model._execute_request({"messages": [], "model": "test"})

    def test_connect_error_maps_to_connection_error(self):
        import httpx

        from workbench.models.errors import ModelConnectionError
        from workbench.models.llamacpp_server import LlamaCppServerModel

        model = LlamaCppServerModel(base_url="http://127.0.0.1:8080")
        model.client = MagicMock()
        model.client.post.side_effect = httpx.ConnectError("refused")

        with pytest.raises(ModelConnectionError):
            model._execute_request({"messages": [], "model": "test"})

    def test_http_status_error_maps_to_response_error(self):
        import httpx

        from workbench.models.errors import ModelResponseError
        from workbench.models.llamacpp_server import LlamaCppServerModel

        model = LlamaCppServerModel(base_url="http://127.0.0.1:8080")
        model.client = MagicMock()

        response = MagicMock(spec=httpx.Response)
        response.status_code = 500
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=response,
        )
        model.client.post.return_value = response

        with pytest.raises(ModelResponseError):
            model._execute_request({"messages": [], "model": "test"})


# ================================================================== #
#  B. SERVER INTEGRATION TESTS (skip if offline)                       #
# ================================================================== #


def _server_available(url: str = "http://127.0.0.1:8080") -> bool:
    """Check if the model server is reachable."""
    try:
        import httpx

        r = httpx.get(f"{url}/v1/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _embedding_server_available(url: str = "http://127.0.0.1:8081") -> bool:
    """Check if the Infinity embedding server is reachable."""
    try:
        import httpx

        r = httpx.get(f"{url}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _reranker_server_available(url: str = "http://127.0.0.1:8082") -> bool:
    """Check if the Infinity reranker server is reachable."""
    try:
        import httpx

        r = httpx.get(f"{url}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _server_available(), reason="Model server not running")
class TestModelServerIntegration:
    """Live tests against the running model server."""

    def _make_model(self):
        from workbench.models.llamacpp_server import LlamaCppServerModel

        return LlamaCppServerModel(
            base_url="http://127.0.0.1:8080",
            timeout_seconds=30,
        )

    def test_single_turn_roundtrip(self):
        """Basic single-turn completion works."""
        model = self._make_model()
        result = model.generate(
            [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
            temperature=0.1,
            max_tokens=10,
        )
        assert result.text.strip()
        assert result.tokens_out > 0

    def test_multi_turn_roundtrip(self):
        """Multi-turn messages are accepted and produce coherent output."""
        model = self._make_model()
        result = model.generate(
            MULTI_TURN_MESSAGES,
            temperature=0.3,
            max_tokens=100,
        )
        assert result.text.strip()
        assert result.tokens_in > 0

    def test_system_prompt_respected(self):
        """System prompt should influence the response."""
        model = self._make_model()
        result = model.generate(
            [
                {"role": "system", "content": "You are a pirate. Always say 'Arrr'."},
                {"role": "user", "content": "Hello!"},
            ],
            temperature=0.3,
            max_tokens=50,
        )
        # Loose check — model should at least produce text
        assert len(result.text.strip()) > 0

    def test_longer_conversation_history(self):
        """Server should handle 8+ turn conversations without error."""
        model = self._make_model()
        messages = [{"role": "system", "content": "You are helpful."}]
        for i in range(4):
            messages.append(
                {"role": "user", "content": f"Question {i}: What is {i}+{i}?"}
            )
            messages.append({"role": "assistant", "content": f"Answer: {i * 2}"})
        messages.append({"role": "user", "content": "What were all the answers?"})

        result = model.generate(messages, temperature=0.3, max_tokens=100)
        assert result.text.strip()

    def test_usage_fields_present_or_gracefully_absent(self):
        """Verify we get usage data or handle its absence."""
        model = self._make_model()
        result = model.generate(
            [{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
        # Should have non-negative token counts regardless
        assert result.tokens_in >= 0
        assert result.tokens_out >= 0

    def test_server_matches_openai_chat_shape(self):
        """Raw response should have choices[0].message.content."""
        import httpx

        resp = httpx.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]


# ================================================================== #
#  C. INFINITY ENDPOINT TESTS (embedding + reranker)                   #
# ================================================================== #


@pytest.mark.integration
@pytest.mark.skipif(
    not _embedding_server_available(),
    reason="Infinity embedding server not running at :8081",
)
class TestInfinityEmbeddingEndpoint:
    """
    Contract tests for the Infinity embedding server.

    Verifies the HTTP API shape that HTTPEmbedder depends on.
    """

    def test_health_endpoint(self):
        import httpx

        resp = httpx.get("http://127.0.0.1:8081/health", timeout=5)
        assert resp.status_code == 200

    def test_embed_single_text(self):
        """Single text embedding returns expected shape."""
        import httpx

        resp = httpx.post(
            "http://127.0.0.1:8081/embeddings",
            json={
                "model": "intfloat/multilingual-e5-base",
                "input": ["Battle of Stalingrad overview"],
            },
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) == 1
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        # multilingual-e5-base produces 768-dim vectors
        assert len(embedding) == 768

    def test_embed_batch(self):
        """Batch embedding returns one result per input."""
        import httpx

        texts = ["first query", "second query", "third query"]
        resp = httpx.post(
            "http://127.0.0.1:8081/embeddings",
            json={
                "model": "intfloat/multilingual-e5-base",
                "input": texts,
            },
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == len(texts)

    def test_embedding_dimensions_consistent(self):
        """All embeddings from same model should have same dimensionality."""
        import httpx

        texts = ["short", "a longer text about something specific"]
        resp = httpx.post(
            "http://127.0.0.1:8081/embeddings",
            json={"model": "intfloat/multilingual-e5-base", "input": texts},
            timeout=10,
        )
        data = resp.json()
        dims = [len(item["embedding"]) for item in data["data"]]
        assert dims[0] == dims[1]


@pytest.mark.integration
@pytest.mark.skipif(
    not _reranker_server_available(),
    reason="Infinity reranker server not running at :8082",
)
class TestInfinityRerankerEndpoint:
    """
    Contract tests for the Infinity reranker server.

    Verifies the HTTP API shape that HTTPReranker depends on.
    """

    def test_health_endpoint(self):
        import httpx

        resp = httpx.get("http://127.0.0.1:8082/health", timeout=5)
        assert resp.status_code == 200

    def test_rerank_basic(self):
        """Reranker should accept query + documents and return scores."""
        import httpx

        resp = httpx.post(
            "http://127.0.0.1:8082/rerank",
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": "Battle of Stalingrad",
                "documents": [
                    "The Battle of Stalingrad was a major battle of World War II.",
                    "Photosynthesis is the process by which plants convert light.",
                    "Stalingrad was fought between August 1942 and February 1943.",
                ],
            },
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        results = data["results"]
        assert len(results) == 3

        # Each result should have index and relevance_score
        for r in results:
            assert "index" in r
            assert "relevance_score" in r
            assert isinstance(r["relevance_score"], (int, float))

    def test_rerank_ordering(self):
        """Relevant documents should score higher than irrelevant ones."""
        import httpx

        resp = httpx.post(
            "http://127.0.0.1:8082/rerank",
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": "Battle of Stalingrad",
                "documents": [
                    "The Battle of Stalingrad was the bloodiest battle of WW2.",
                    "Python is a programming language created by Guido van Rossum.",
                ],
            },
            timeout=10,
        )
        data = resp.json()
        results = sorted(
            data["results"], key=lambda x: x["relevance_score"], reverse=True
        )
        # The Stalingrad document should rank higher
        assert results[0]["index"] == 0
