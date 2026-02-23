"""
Adapter for llama-cpp-python's OpenAI-compatible server.

Connects to a local llama.cpp server and implements the LanguageModel interface.
"""

import time
from typing import Any

import httpx

from workbench.core.registry import register_component
from workbench.core.types import ModelResponse
from workbench.models.errors import (
    ModelConnectionError,
    ModelResponseError,
    ModelTimeoutError,
)
from workbench.models.language_model import (
    GenerationSettings,
    extract_settings,
    validate_messages,
)
from workbench.models.token_counting import count_tokens_from_usage, estimate_tokens
from workbench.observability.logging import get_logger
from workbench.observability.tracing import start_span


class LlamaCppServerModel:
    """
    Language model adapter for llama.cpp's OpenAI-compatible server.

    This adapter connects to a local llama-server instance:
        bash scripts/start_model_server.sh

    The server provides an OpenAI-compatible API at /v1/chat/completions.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model_name: str = "local-model",
        timeout_seconds: float = 60.0,
        default_temperature: float = 0.7,
        default_max_tokens: int = 2048,
        **kwargs: Any,
    ) -> None:
        """
        Initialize llama.cpp server adapter.

        Args:
            base_url: Base URL of llama.cpp server (default: http://127.0.0.1:8080)
            model_name: Model identifier for logging (default: "local-model")
            timeout_seconds: Request timeout in seconds (default: 60)
            default_temperature: Default temperature for generation
            default_max_tokens: Default max tokens for generation
            **kwargs: Additional configuration (ignored)
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

        # Default generation settings
        self.default_settings = GenerationSettings(
            temperature=default_temperature,
            max_tokens=default_max_tokens,
        )

        # Create HTTP client
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
        )

        # Lazy logger initialization (only create when needed)
        self._logger = None

    @property
    def logger(self):
        """Lazy logger initialization - only creates logger when first accessed."""
        if self._logger is None:
            try:
                self._logger = get_logger()
            except RuntimeError:
                # No run context - skip logging
                # This allows the model to work standalone without full observability setup
                pass
        return self._logger

    def _log_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Log an event if logger is available."""
        if self.logger is not None:
            self.logger.log_event(event_name, data)

    def generate(
        self,
        messages: list[dict[str, str]],
        **settings: Any,
    ) -> ModelResponse:
        """
        Generate text from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **settings: Generation settings (temperature, max_tokens, etc.)

        Returns:
            ModelResponse with generated text and metadata

        Raises:
            ModelConnectionError: If unable to connect to server
            ModelTimeoutError: If request times out
            ModelResponseError: If server returns an error
        """
        # Validate messages
        validate_messages(messages)

        # Extract settings
        gen_settings = extract_settings(settings, defaults=self.default_settings)

        # Prepare request
        request_data = self._create_request(messages, gen_settings)

        # Execute with tracing
        with start_span(
            "model.generate",
            attributes={
                "model.provider": "llamacpp_server",
                "model.name": self.model_name,
                "model.temperature": gen_settings.temperature,
                "model.max_tokens": gen_settings.max_tokens,
                "messages.count": len(messages),
            },
        ):
            return self._execute_request(request_data)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        For llama.cpp server, we use a simple estimation since
        the server doesn't provide a dedicated tokenization endpoint.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count

        Note:
            This is a rough estimate. For exact counts, you would need
            to load the tokenizer separately or use the model's tokenizer.
        """
        return estimate_tokens(text)

    def _create_request(
        self,
        messages: list[dict[str, str]],
        settings: GenerationSettings,
    ) -> dict[str, Any]:
        """
        Create request payload for the server.

        Args:
            messages: Message list
            settings: Generation settings

        Returns:
            Request dictionary
        """
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "stream": settings.stream,
        }

        # Add optional parameters if set
        if settings.top_p is not None:
            request["top_p"] = settings.top_p
        if settings.top_k is not None:
            request["top_k"] = settings.top_k
        if settings.stop is not None:
            request["stop"] = settings.stop

        return request

    def _execute_request(self, request_data: dict[str, Any]) -> ModelResponse:
        """
        Execute the API request and parse response.

        Args:
            request_data: Request payload

        Returns:
            ModelResponse

        Raises:
            ModelConnectionError: If unable to connect
            ModelTimeoutError: If request times out
            ModelResponseError: If response is invalid
        """
        start_time = time.time()

        try:
            # Make request to /v1/chat/completions
            response = self.client.post(
                "/v1/chat/completions",
                json=request_data,
            )
            response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            data = response.json()
            return self._parse_response(data, latency_ms)

        except httpx.TimeoutException as e:
            self._log_event(
                "model_timeout",
                {
                    "model": self.model_name,
                    "timeout_seconds": self.timeout_seconds,
                    "error": str(e),
                },
            )
            raise ModelTimeoutError(
                f"Request timed out after {self.timeout_seconds}s"
            ) from e

        except httpx.ConnectError as e:
            self._log_event(
                "model_connection_error",
                {
                    "model": self.model_name,
                    "base_url": self.base_url,
                    "error": str(e),
                },
            )
            raise ModelConnectionError(
                f"Unable to connect to model server at {self.base_url}. "
                "Make sure the server is running with: "
                "bash scripts/start_model_server.sh"
            ) from e

        except httpx.HTTPStatusError as e:
            self._log_event(
                "model_http_error",
                {
                    "model": self.model_name,
                    "status_code": e.response.status_code,
                    "error": str(e),
                },
            )
            raise ModelResponseError(
                f"Server returned error status: {e.response.status_code}"
            ) from e

        except Exception as e:
            self._log_event(
                "model_error",
                {
                    "model": self.model_name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            raise ModelResponseError(f"Unexpected error: {e}") from e

    def _parse_response(self, data: dict[str, Any], latency_ms: float) -> ModelResponse:
        """
        Parse API response into ModelResponse.

        Args:
            data: Response JSON data
            latency_ms: Request latency in milliseconds

        Returns:
            ModelResponse

        Raises:
            ModelResponseError: If response format is invalid
        """
        try:
            # Extract message content
            choices = data.get("choices", [])
            if not choices:
                raise ModelResponseError("No choices in response")

            message = choices[0].get("message", {})
            text = message.get("content", "")

            # Extract finish reason
            finish_reason = choices[0].get("finish_reason", None)

            # Extract token counts from usage
            usage = data.get("usage", {})
            tokens_in, tokens_out = count_tokens_from_usage(
                usage, fallback_text=text, logger=self.logger
            )

            # Log generation event
            self._log_event(
                "model_generated",
                {
                    "model": self.model_name,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "latency_ms": latency_ms,
                    "finish_reason": finish_reason,
                },
            )

            return ModelResponse(
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                metadata={
                    "model": self.model_name,
                    "provider": "llamacpp_server",
                    "raw_usage": usage,
                },
            )

        except KeyError as e:
            raise ModelResponseError(f"Invalid response format: missing {e}") from e

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


@register_component(
    name="llamacpp_server",
    component_type="language_model",
    description="llama.cpp server adapter (OpenAI-compatible API)",
)
def build_llamacpp_server_model(**config: Any) -> LlamaCppServerModel:
    """
    Build LlamaCppServerModel from configuration.

    Args:
        **config: Configuration parameters

    Returns:
        LlamaCppServerModel instance
    """
    return LlamaCppServerModel(**config)
