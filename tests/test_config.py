"""Tests for config loading and validation."""

from pathlib import Path

import pytest

from tonecraft.config import load_config
from tonecraft.schemas import ProjectConfig


def test_load_valid_config(sample_config_file: Path):
    config = load_config(sample_config_file)
    assert isinstance(config, ProjectConfig)
    assert config.target.slm == "microsoft/phi-3-mini"
    assert config.generation.provider == "openai"
    assert config.output.output_dir == "./output"


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="tonecraft.toml"):
        load_config(tmp_path / "nonexistent.toml")


def test_missing_target_slm_raises(tmp_path: Path):
    config_file = tmp_path / "tonecraft.toml"
    config_file.write_text("""\
[target]
slm_context_window = 4096
training_format = "chatml"

[generation]
provider = "openai"
model_expert = "gpt-4o"
model_agent = "gpt-4o-mini"
""")
    with pytest.raises(Exception):
        load_config(config_file)


def test_default_values(tmp_path: Path):
    config_file = tmp_path / "tonecraft.toml"
    config_file.write_text("""\
[target]
slm = "microsoft/phi-3-mini"
""")
    config = load_config(config_file)
    assert config.generation.provider == "openai"
    assert config.generation.max_pairs == 500
    assert config.generation.max_iterations == 3
    assert config.generation.confidence_threshold == 0.7
    assert config.output.output_dir == "./output"


def test_invalid_provider_raises(tmp_path: Path):
    config_file = tmp_path / "tonecraft.toml"
    config_file.write_text("""\
[target]
slm = "microsoft/phi-3-mini"

[generation]
provider = "unknown_provider"
""")
    with pytest.raises(Exception):
        load_config(config_file)


def test_load_config_accepts_string_path(sample_config_file: Path):
    config = load_config(str(sample_config_file))
    assert isinstance(config, ProjectConfig)
