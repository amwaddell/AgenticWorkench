"""
Exception classes for model adapters.
"""


class ModelError(Exception):
    """Base exception for model-related errors."""

    pass


class ModelTimeoutError(ModelError):
    """Raised when a model request times out."""

    pass


class ModelConnectionError(ModelError):
    """Raised when unable to connect to model server."""

    pass


class ModelResponseError(ModelError):
    """Raised when model returns an invalid or error response."""

    pass
