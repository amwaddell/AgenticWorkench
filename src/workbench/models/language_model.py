"""
Base language model utilities and data structures.

Provides shared types and helper functions for all language model adapters.
"""

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")

    model_config = {"frozen": True}


class GenerationSettings(BaseModel):
    """Settings for text generation."""

    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=2048, ge=1, description="Maximum tokens to generate"
    )
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling threshold"
    )
    top_k: int | None = Field(default=None, ge=0, description="Top-K sampling")
    stop: list[str] | None = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Enable streaming responses")

    model_config = {"frozen": True}


def messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """
    Convert ChatMessage objects to dictionary format.

    Args:
        messages: List of ChatMessage objects

    Returns:
        List of message dictionaries
    """
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def dicts_to_messages(dicts: list[dict[str, str]]) -> list[ChatMessage]:
    """
    Convert message dictionaries to ChatMessage objects.

    Args:
        dicts: List of message dictionaries

    Returns:
        List of ChatMessage objects
    """
    return [ChatMessage(role=d["role"], content=d["content"]) for d in dicts]


def validate_messages(messages: list[dict[str, str]]) -> None:
    """
    Validate message format.

    Args:
        messages: List of message dictionaries

    Raises:
        ValueError: If messages are invalid
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Message {i} must be a dictionary")
        if "role" not in msg:
            raise ValueError(f"Message {i} missing 'role' field")
        if "content" not in msg:
            raise ValueError(f"Message {i} missing 'content' field")
        if msg["role"] not in ("system", "user", "assistant"):
            raise ValueError(
                f"Message {i} has invalid role: {msg['role']}. "
                "Must be 'system', 'user', or 'assistant'"
            )


def create_chat_messages(
    system: str | None = None,
    user: str | None = None,
    assistant: str | None = None,
) -> list[ChatMessage]:
    """
    Convenience function to create a simple chat message list.

    Args:
        system: Optional system message
        user: Optional user message
        assistant: Optional assistant message

    Returns:
        List of ChatMessage objects
    """
    messages = []
    if system:
        messages.append(ChatMessage(role="system", content=system))
    if user:
        messages.append(ChatMessage(role="user", content=user))
    if assistant:
        messages.append(ChatMessage(role="assistant", content=assistant))
    return messages


def extract_settings(
    settings_dict: dict[str, Any],
    defaults: GenerationSettings | None = None,
) -> GenerationSettings:
    """
    Extract generation settings from a dictionary.

    Args:
        settings_dict: Dictionary of settings
        defaults: Optional default settings to merge with

    Returns:
        GenerationSettings object
    """
    if defaults:
        # Merge with defaults
        merged = defaults.model_dump()
        merged.update(settings_dict)
        return GenerationSettings(**merged)
    else:
        return GenerationSettings(**settings_dict)
