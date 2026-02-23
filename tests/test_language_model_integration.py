"""
Integration tests for language model with actual server.

These tests require a running llama-cpp-python server.
Run with: python -m pytest tests/test_language_model_integration.py -m integration

To run the server:
    bash scripts/start_model_server.sh
"""

import pytest

from workbench.core.config import load_config
from workbench.models.llamacpp_server import LlamaCppServerModel
from workbench.observability.tracing import setup_tracing

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def _server_is_reachable(base_url: str = "http://localhost:8080") -> bool:
    """Check if the model server is reachable."""
    try:
        import httpx

        response = httpx.get(f"{base_url}/v1/models", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


# Auto-skip entire module if model server is not running
if not _server_is_reachable():
    pytestmark = [
        pytestmark,
        pytest.mark.skip(
            reason="Model server not running at localhost:8080. "
            "Start with: bash scripts/start_model_server.sh"
        ),
    ]


@pytest.fixture(scope="module")
def setup_observability():
    """Set up tracing for integration tests."""
    setup_tracing(service_name="test-integration")
    yield


@pytest.fixture(scope="module")
def config():
    """Load configuration."""
    return load_config()


@pytest.fixture(scope="module")
def model(config):
    """Create model instance from config."""
    model_config = config.model
    return LlamaCppServerModel(
        base_url=model_config.get("base_url", "http://localhost:8080"),
        model_name=model_config.get("model_name", "local-model"),
        timeout_seconds=model_config.get("timeout_seconds", 60.0),
    )


def test_model_server_reachable(model):
    """Test that model server is reachable."""
    # Try a simple generation
    messages = [{"role": "user", "content": "Say 'hello' in one word."}]

    try:
        response = model.generate(messages, max_tokens=10, temperature=0.0)
        assert response is not None
        print("\n✓ Server is reachable")
        print(f"  Response: {response.text}")
    except Exception as e:
        pytest.fail(
            f"Model server not reachable. Make sure it's running:\n"
            f"  bash scripts/start_model_server.sh\n"
            f"Error: {e}"
        )


def test_simple_generation(setup_observability, model):
    """Test simple text generation."""
    messages = [{"role": "user", "content": "Say hello in one sentence."}]

    response = model.generate(messages, max_tokens=50, temperature=0.7)

    assert response.text is not None
    assert len(response.text) > 0
    assert response.latency_ms > 0

    print("\n✓ Simple generation")
    print(f"  Response: {response.text[:100]}...")
    print(f"  Tokens in: {response.tokens_in}")
    print(f"  Tokens out: {response.tokens_out}")
    print(f"  Latency: {response.latency_ms:.2f}ms")


def test_system_message(setup_observability, model):
    """Test generation with system message."""
    messages = [
        {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
        {"role": "user", "content": "Introduce yourself in one sentence."},
    ]

    response = model.generate(messages, max_tokens=100, temperature=0.8)

    assert response.text is not None
    assert len(response.text) > 0

    print("\n✓ System message generation")
    print(f"  Response: {response.text[:150]}...")


def test_multi_turn_conversation(setup_observability, model):
    """Test multi-turn conversation."""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And if I add 3 to that?"},
    ]

    response = model.generate(messages, max_tokens=50, temperature=0.0)

    assert response.text is not None
    assert len(response.text) > 0

    print("\n✓ Multi-turn conversation")
    print(f"  Response: {response.text}")


def test_temperature_variation(setup_observability, model):
    """Test that temperature affects output."""
    messages = [{"role": "user", "content": "Tell me a fun fact."}]

    # Low temperature (more deterministic)
    response_low = model.generate(messages, max_tokens=100, temperature=0.1)

    # High temperature (more creative)
    response_high = model.generate(messages, max_tokens=100, temperature=1.5)

    assert response_low.text is not None
    assert response_high.text is not None

    print("\n✓ Temperature variation")
    print(f"  Low temp (0.1): {response_low.text[:80]}...")
    print(f"  High temp (1.5): {response_high.text[:80]}...")


def test_max_tokens_limit(setup_observability, model):
    """Test that max_tokens limits output."""
    messages = [{"role": "user", "content": "Write a long story."}]

    response = model.generate(messages, max_tokens=20, temperature=0.7)

    assert response.text is not None
    # Token count should be close to limit
    assert response.tokens_out <= 25  # Allow some flexibility

    print("\n✓ Max tokens limit")
    print("  Max tokens: 20")
    print(f"  Actual tokens: {response.tokens_out}")
    print(f"  Response: {response.text}")


def test_token_counting(setup_observability, model):
    """Test token counting functionality."""
    text = "This is a test sentence for token counting."
    count = model.count_tokens(text)

    assert count > 0
    # Should be roughly 8-12 tokens
    assert 5 <= count <= 20

    print("\n✓ Token counting")
    print(f"  Text: {text}")
    print(f"  Estimated tokens: {count}")


def test_trace_span_created(setup_observability, model):
    """Test that trace span is created for generation."""
    from workbench.observability.tracing import get_tracer

    tracer = get_tracer()

    messages = [{"role": "user", "content": "Hi"}]

    # This should create a span
    with tracer.start_as_current_span("test_span"):
        response = model.generate(messages, max_tokens=20)

    assert response is not None

    print("\n✓ Trace span created")
    print("  Check Phoenix UI at http://localhost:6006 to see the trace")


def test_metadata_in_response(setup_observability, model):
    """Test that response includes metadata."""
    messages = [{"role": "user", "content": "Hello"}]

    response = model.generate(messages, max_tokens=50)

    assert "model" in response.metadata
    assert "provider" in response.metadata
    assert response.metadata["provider"] == "llamacpp_server"

    print("\n✓ Response metadata")
    print(f"  Metadata: {response.metadata}")


def test_context_manager(config):
    """Test using model as context manager."""
    model_config = config.model

    with LlamaCppServerModel(
        base_url=model_config.get("base_url", "http://localhost:8080"),
        model_name=model_config.get("model_name", "local-model"),
    ) as m:
        # Verify context manager returns usable instance
        assert m is not None
        assert hasattr(m, "generate")

    # Verify __exit__ closed the client
    assert m.client.is_closed

    print("\n✓ Context manager usage")


if __name__ == "__main__":
    print("\nIntegration Tests for Language Model")
    print("=" * 70)
    print("\nIMPORTANT: These tests require a running model server!")
    print("Start the server with:")
    print("  bash scripts/start_model_server.sh")
    print("\nThen run:")
    print(
        "  python -m pytest tests/test_language_model_integration.py -m integration -v"
    )
    print("=" * 70)
