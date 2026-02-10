"""
Configuration loading and validation.

Loads YAML configs and creates frozen snapshots for reproducibility.
"""

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class WorkbenchConfig(BaseModel):
    """
    Main configuration object with validation.

    This ensures all required config sections are present.
    """

    model_config = ConfigDict(extra="allow", frozen=False)

    model: dict[str, Any] = Field(..., description="Model configuration")
    embeddings: dict[str, Any] = Field(..., description="Embeddings configuration")
    retrieval: dict[str, Any] = Field(..., description="Retrieval configuration")
    reranking: dict[str, Any] = Field(..., description="Reranking configuration")
    chunking: dict[str, Any] = Field(..., description="Chunking configuration")
    paths: dict[str, Any] = Field(..., description="File paths configuration")
    runs: dict[str, Any] = Field(..., description="Run settings")
    observability: dict[str, Any] = Field(..., description="Observability settings")

    def model_post_init(self, __context):
        """Validate configuration after initialization."""
        # Validate paths
        required_path_keys = ["data_dir", "indexes_dir"]
        for key in required_path_keys:
            if key not in self.paths:
                raise ValueError(f"Missing required path config: {key}")

        # Validate model
        required_model_keys = ["provider", "base_url"]
        for key in required_model_keys:
            if key not in self.model:
                raise ValueError(f"Missing required model config: {key}")


def load_yaml_file(path: Path) -> dict[str, Any]:
    """
    Load a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary of config data

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is invalid YAML
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty config file: {path}")

    return data


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two config dictionaries.

    Args:
        base: Base configuration
        override: Override configuration (takes precedence)

    Returns:
        Merged configuration
    """
    merged = copy.deepcopy(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> WorkbenchConfig:
    """
    Load and validate configuration.

    Args:
        config_path: Path to config file (defaults to configs/defaults.yaml)
        overrides: Optional dictionary of overrides to merge in

    Returns:
        Validated WorkbenchConfig object

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config validation fails
    """
    if config_path is None:
        # Default to configs/defaults.yaml relative to project root
        config_path = (
            Path(__file__).parent.parent.parent.parent / "configs" / "defaults.yaml"
        )

    # Load base config
    config_data = load_yaml_file(config_path)

    # Merge overrides if provided
    if overrides:
        config_data = merge_configs(config_data, overrides)

    # Validate and return
    return WorkbenchConfig(**config_data)


def create_config_snapshot(config: WorkbenchConfig) -> dict[str, Any]:
    """
    Create a frozen snapshot of config for reproducibility.

    This snapshot will be saved with run records.

    Args:
        config: WorkbenchConfig object

    Returns:
        Dictionary snapshot of config
    """
    # Convert to dict and deep copy
    snapshot = copy.deepcopy(config.model_dump())
    return snapshot


def save_config_snapshot(snapshot: dict[str, Any], output_path: Path) -> None:
    """
    Save config snapshot to YAML file.

    Args:
        snapshot: Config snapshot dictionary
        output_path: Path to save snapshot
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
