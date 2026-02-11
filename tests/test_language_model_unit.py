"""
Unit tests for language model components.

These tests do not require a running model server.
"""

from unittest.mock import Mock, patch

import httpx

from workbench.core.types import ModelResponse
from workbench.models.errors import (
    ModelConnectionError,
    ModelResponseError,
    ModelTimeoutError,
)
from workbench.models.language_model import (
    ChatMessage,
    GenerationSettings,
    create_chat_messages,
    dicts_to_messages,
    messages_to_dicts,
    validate_messages,
)
from workbench.models.llamacpp_server import LlamaCppServerModel
from workbench.models.token_counting import (
    count_tokens_from_usage,
    estimate_message_tokens,
    estimate_tokens,
)


def test_chat_message_creation():
    """Test ChatMessage creation and validation."""
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_generation_settings_defaults():
    """Test GenerationSettings with defaults."""
    settings = GenerationSettings()
    assert settings.temperature == 0.7
    assert settings.max_tokens == 2048
    assert settings.stream is False


def test_generation_settings_custom():
    """Test GenerationSettings with custom values."""
    settings = GenerationSettings(temperature=0.5, max_tokens=1000, top_p=0.9)
    assert settings.temperature == 0.5
    assert settings.max_tokens == 1000
    assert settings.top_p == 0.9


def test_messages_to_dicts():
    """Test converting ChatMessage objects to dictionaries."""
    messages = [
        ChatMessage(role="system", content="You are helpful"),
        ChatMessage(role="user", content="Hello"),
    ]
    dicts = messages_to_dicts(messages)
    assert len(dicts) == 2
    assert dicts[0] == {"role": "system", "content": "You are helpful"}
    assert dicts[1] == {"role": "user", "content": "Hello"}


def test_dicts_to_messages():
    """Test converting dictionaries to ChatMessage objects."""
    dicts = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    messages = dicts_to_messages(dicts)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
    assert messages[1].role == "assistant"


def test_create_chat_messages():
    """Test convenience function for creating messages."""
    messages = create_chat_messages(system="Be helpful", user="Hello", assistant="Hi")
    assert len(messages) == 3
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[2].role == "assistant"


def test_validate_messages_success():
    """Test message validation with valid messages."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    validate_messages(messages)  # Should not raise


def test_validate_messages_empty():
    """Test message validation with empty list."""
    try:
        validate_messages([])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot be empty" in str(e)


def test_validate_messages_missing_role():
    """Test message validation with missing role."""
    try:
        validate_messages([{"content": "Hello"}])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "missing 'role'" in str(e)


def test_validate_messages_invalid_role():
    """Test message validation with invalid role."""
    try:
        validate_messages([{"role": "invalid", "content": "Hello"}])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid role" in str(e)


def test_estimate_tokens():
    """Test token estimation."""
    text = "This is a test sentence."
    tokens = estimate_tokens(text)
    # Should be roughly 6-7 tokens
    assert 5 <= tokens <= 10


def test_estimate_tokens_empty():
    """Test token estimation with empty text."""
    assert estimate_tokens("") == 0


def test_estimate_message_tokens():
    """Test token estimation for messages."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there, how are you?"},
    ]
    tokens = estimate_message_tokens(messages)
    # Should be positive
    assert tokens > 0


def test_count_tokens_from_usage_with_data():
    """Test token counting with complete usage data."""
    usage = {"prompt_tokens": 10, "completion_tokens": 20}
    tokens_in, tokens_out = count_tokens_from_usage(usage)
    assert tokens_in == 10
    assert tokens_out == 20


def test_count_tokens_from_usage_missing_data():
    """Test token counting with missing usage data."""
    usage = {}
    tokens_in, tokens_out = count_tokens_from_usage(
        usage, fallback_text="test response"
    )
    # Should use estimation
    assert tokens_in == 0  # No prompt info
    assert tokens_out > 0  # Estimated from text


def test_count_tokens_from_usage_partial_data():
    """Test token counting with partial usage data."""
    usage = {"prompt_tokens": 15}
    tokens_in, tokens_out = count_tokens_from_usage(
        usage, fallback_text="test response"
    )
    assert tokens_in == 15  # From usage
    assert tokens_out > 0  # Estimated from text


def test_llamacpp_server_init():
    """Test LlamaCppServerModel initialization."""
    model = LlamaCppServerModel(
        base_url="http://localhost:8080",
        model_name="test-model",
        timeout_seconds=30.0,
    )
    assert model.base_url == "http://localhost:8080"
    assert model.model_name == "test-model"
    assert model.timeout_seconds == 30.0


def test_llamacpp_server_create_request():
    """Test request creation."""
    model = LlamaCppServerModel()
    settings = GenerationSettings(temperature=0.5, max_tokens=100)
    messages = [{"role": "user", "content": "Hello"}]

    request = model._create_request(messages, settings)

    assert request["model"] == model.model_name
    assert request["messages"] == messages
    assert request["temperature"] == 0.5
    assert request["max_tokens"] == 100


def test_llamacpp_server_parse_response():
    """Test response parsing."""
    model = LlamaCppServerModel()

    response_data = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }

    result = model._parse_response(response_data, latency_ms=100.0)

    assert isinstance(result, ModelResponse)
    assert result.text == "Hello there!"
    assert result.tokens_in == 5
    assert result.tokens_out == 3
    assert result.latency_ms == 100.0
    assert result.finish_reason == "stop"


def test_llamacpp_server_parse_response_no_usage():
    """Test response parsing without usage data."""
    model = LlamaCppServerModel()

    response_data = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }

    result = model._parse_response(response_data, latency_ms=50.0)

    assert result.text == "Test response"
    # Should use estimation
    assert result.tokens_out > 0


def test_llamacpp_server_parse_response_invalid():
    """Test response parsing with invalid data."""
    model = LlamaCppServerModel()

    try:
        model._parse_response({"invalid": "data"}, latency_ms=0)
        assert False, "Should have raised ModelResponseError"
    except ModelResponseError:
        pass


@patch("httpx.Client.post")
def test_llamacpp_server_generate_success(mock_post):
    """Test successful generation with mocked HTTP client."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    mock_post.return_value = mock_response

    model = LlamaCppServerModel()
    messages = [{"role": "user", "content": "Hi"}]

    result = model.generate(messages)

    assert result.text == "Hello!"
    assert result.tokens_in == 5
    assert result.tokens_out == 2


@patch("httpx.Client.post")
def test_llamacpp_server_generate_timeout(mock_post):
    """Test generation with timeout."""
    mock_post.side_effect = httpx.TimeoutException("Timeout")

    model = LlamaCppServerModel(timeout_seconds=1.0)
    messages = [{"role": "user", "content": "Hi"}]

    try:
        model.generate(messages)
        assert False, "Should have raised ModelTimeoutError"
    except ModelTimeoutError:
        pass


@patch("httpx.Client.post")
def test_llamacpp_server_generate_connection_error(mock_post):
    """Test generation with connection error."""
    mock_post.side_effect = httpx.ConnectError("Connection failed")

    model = LlamaCppServerModel()
    messages = [{"role": "user", "content": "Hi"}]

    try:
        model.generate(messages)
        assert False, "Should have raised ModelConnectionError"
    except ModelConnectionError:
        pass


def test_llamacpp_server_count_tokens():
    """Test token counting."""
    model = LlamaCppServerModel()
    count = model.count_tokens("This is a test sentence.")
    assert count > 0


def test_llamacpp_server_context_manager():
    """Test using model as context manager."""
    with LlamaCppServerModel() as model:
        assert model is not None
    # Should close client after exiting context


if __name__ == "__main__":
    print("Running unit tests...")
    print("=" * 70)

    # Run all test functions
    test_functions = [
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: {e}")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        exit(1)
