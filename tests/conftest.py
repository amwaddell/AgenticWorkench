"""
Shared fixtures and markers for Day 9 chat + server tests.

Usage:
    pytest tests/ -m "not integration"     # unit tests only
    pytest tests/ -m integration            # server tests only
    pytest tests/                            # everything
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ------------------------------------------------------------------ #
#  Tracing / observability stubs                                       #
# ------------------------------------------------------------------ #
#  Graph nodes call start_span / add_span_attributes.  In unit tests
#  we don't need real tracing, so we patch them globally.


@pytest.fixture(autouse=True)
def _mock_tracing(monkeypatch):
    """Disable tracing in all tests unless explicitly testing tracing."""
    try:
        import workbench.observability.tracing as tracing_mod

        monkeypatch.setattr(
            tracing_mod,
            "start_span",
            MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            ),
        )
        monkeypatch.setattr(tracing_mod, "add_span_attributes", MagicMock())
    except ImportError:
        pass  # tracing module not installed — nothing to mock


# ------------------------------------------------------------------ #
#  Pytest markers                                                      #
# ------------------------------------------------------------------ #


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: tests that require running servers"
    )
