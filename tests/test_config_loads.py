"""
Tests for configuration loading and validation.
"""

from pathlib import Path

import pytest

from workbench.core.config import (
    WorkbenchConfig,
    create_config_snapshot,
    load_config,
    load_yaml_file,
    merge_configs,
)


def test_load_defaults_yaml():
    """Test that defaults.yaml loads without errors."""
    config_path = Path(__file__).parent.parent / "configs" / "defaults.yaml"
    data = load_yaml_file(config_path)

    # Check that major sections exist
    assert "model" in data
    assert "embeddings" in data
    assert "retrieval" in data
    assert "reranking" in data
    assert "paths" in data


def test_load_config_validates():
    """Test that config validation catches missing required fields."""
    config = load_config()

    # Should be a WorkbenchConfig instance
    assert isinstance(config, WorkbenchConfig)

    # Check required sections exist
    assert config.model is not None
    assert config.embeddings is not None
    assert config.retrieval is not None
    assert config.paths is not None


def test_config_has_required_model_fields():
    """Test that model config has required fields."""
    config = load_config()

    assert "provider" in config.model
    assert "base_url" in config.model


def test_config_has_required_path_fields():
    """Test that paths config has required fields."""
    config = load_config()

    assert "data_dir" in config.paths
    assert "indexes_dir" in config.paths


def test_merge_configs_simple():
    """Test simple config merging."""
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}

    merged = merge_configs(base, override)

    assert merged["a"] == 1
    assert merged["b"] == 3  # Override wins
    assert merged["c"] == 4


def test_merge_configs_nested():
    """Test nested config merging."""
    base = {
        "model": {"provider": "base", "temperature": 0.5},
        "paths": {"data_dir": "./data"},
    }
    override = {
        "model": {"temperature": 0.7},
    }

    merged = merge_configs(base, override)

    # Provider should remain from base
    assert merged["model"]["provider"] == "base"
    # Temperature should be overridden
    assert merged["model"]["temperature"] == 0.7
    # Paths should remain unchanged
    assert merged["paths"]["data_dir"] == "./data"


def test_load_config_with_overrides():
    """Test loading config with runtime overrides."""
    overrides = {
        "model": {"temperature": 0.9},
    }

    config = load_config(overrides=overrides)

    assert config.model["temperature"] == 0.9
    # Other fields should still exist
    assert "provider" in config.model


def test_create_config_snapshot():
    """Test creating a frozen config snapshot."""
    config = load_config()
    snapshot = create_config_snapshot(config)

    # Should be a dictionary
    assert isinstance(snapshot, dict)

    # Should have all major sections
    assert "model" in snapshot
    assert "embeddings" in snapshot
    assert "retrieval" in snapshot

    # Should be a deep copy (modifying snapshot doesn't affect config)
    snapshot["model"]["temperature"] = 999
    assert config.model["temperature"] != 999


def test_config_validation_fails_on_missing_required():
    """Test that validation fails when required fields are missing."""
    with pytest.raises(ValueError, match="Missing required"):
        WorkbenchConfig(
            model={},  # Missing required fields
            embeddings={},
            retrieval={},
            reranking={},
            chunking={},
            paths={},  # Missing required fields
            runs={},
            observability={},
        )


def test_load_nonexistent_config():
    """Test that loading a nonexistent config raises FileNotFoundError."""
    fake_path = Path("/nonexistent/config.yaml")

    with pytest.raises(FileNotFoundError):
        load_config(config_path=fake_path)
