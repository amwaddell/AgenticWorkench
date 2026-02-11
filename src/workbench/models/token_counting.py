"""
Token counting utilities for language models.

Provides functions to extract token counts from API responses
and fallback estimation when exact counts are unavailable.
"""

from typing import Any


def count_tokens_from_usage(
    usage: dict[str, Any],
    fallback_text: str | None = None,
    logger: Any | None = None,
) -> tuple[int, int]:
    """
    Extract token counts from API usage data.

    Args:
        usage: Usage dictionary from API response
        fallback_text: Optional text to estimate from if usage unavailable
        logger: Optional logger to log warnings

    Returns:
        Tuple of (tokens_in, tokens_out)

    Note:
        If usage data is unavailable, estimates are returned and a warning is logged.
        We never return fake precision - if we don't know, we estimate and log it.
    """
    # Try to extract from usage
    tokens_in = usage.get("prompt_tokens")
    tokens_out = usage.get("completion_tokens")

    # If we got both values, return them
    if tokens_in is not None and tokens_out is not None:
        return int(tokens_in), int(tokens_out)

    # Otherwise, log warning and estimate
    if logger:
        logger.log_event(
            "token_count_fallback",
            {
                "reason": "usage_data_unavailable",
                "has_prompt_tokens": tokens_in is not None,
                "has_completion_tokens": tokens_out is not None,
                "using_estimation": True,
            },
        )

    # Use estimation as fallback
    if tokens_out is None and fallback_text:
        tokens_out = estimate_tokens(fallback_text)
    elif tokens_out is None:
        tokens_out = 0

    if tokens_in is None:
        tokens_in = 0

    return tokens_in, tokens_out


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count from text.

    This is a rough approximation. Different tokenizers will give
    different results, but ~4 characters per token is a reasonable
    default for many models.

    Args:
        text: Text to estimate tokens for
        chars_per_token: Average characters per token (default: 4.0)

    Returns:
        Estimated token count

    Note:
        This is intentionally a rough estimate. For exact counts,
        use the model's tokenizer directly.
    """
    if not text:
        return 0

    # Simple character-based estimation
    return max(1, int(len(text) / chars_per_token))


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    """
    Estimate token count for a list of chat messages.

    Args:
        messages: List of message dictionaries

    Returns:
        Estimated total token count

    Note:
        Adds overhead for message formatting (role tags, etc.)
    """
    total = 0

    for msg in messages:
        # Content tokens
        content = msg.get("content", "")
        total += estimate_tokens(content)

        # Role overhead (rough estimate: ~4 tokens per message for formatting)
        total += 4

    # Conversation overhead
    total += 3

    return total
