"""Shared fixtures for tonecraft tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


SAMPLE_MD_CONTENT = """\
# Domain

Restaurant service — casual Italian trattoria with a focus on fresh pasta and local wines.

# Roles

**Questioner**: Curious restaurant guest with questions about the menu and daily specials.

**Responder**: Friendly waiter who knows the menu well and offers helpful recommendations.

# Tone Guidelines

- Responder: warm, casual, knowledgeable. Uses "sure", "absolutely". No stiff formality.
- Questioner: direct and curious, may ask about ingredients or preparation.

# Topics

- Pasta dishes and sauces
- Daily specials
- Wine and beverage options
- Vegetarian and vegan options

# Constraints

- Stay in character at all times
- Wine recommendations should name specific varietals
"""


@pytest.fixture
def sample_md_content() -> str:
    return SAMPLE_MD_CONTENT


@pytest.fixture
def sample_md_file(tmp_path: Path, sample_md_content: str) -> Path:
    md_file = tmp_path / "test_domain.md"
    md_file.write_text(sample_md_content)
    return md_file


@pytest.fixture
def sample_config_dict() -> dict:
    return {
        "target": {
            "slm": "microsoft/phi-3-mini",
            "slm_context_window": 4096,
            "training_format": "chatml",
        },
        "generation": {
            "provider": "openai",
            "model_expert": "gpt-4o",
            "model_agent": "gpt-4o-mini",
            "base_url": "",
            "max_pairs": 100,
            "max_iterations": 3,
            "confidence_threshold": 0.7,
        },
        "output": {
            "format": ["jsonl", "json"],
            "output_dir": "./output",
            "include_metadata": True,
        },
    }


@pytest.fixture
def sample_config_file(tmp_path: Path, sample_config_dict: dict) -> Path:
    import tomllib

    content = """\
[target]
slm = "microsoft/phi-3-mini"
slm_context_window = 4096
training_format = "chatml"

[generation]
provider = "openai"
model_expert = "gpt-4o"
model_agent = "gpt-4o-mini"
base_url = ""
max_pairs = 100
max_iterations = 3
confidence_threshold = 0.7

[output]
format = ["jsonl", "json"]
output_dir = "./output"
include_metadata = true
"""
    config_file = tmp_path / "tonecraft.toml"
    config_file.write_text(content)
    return config_file


@pytest.fixture
def mock_instructor_client() -> MagicMock:
    """Provider-agnostic mock instructor client returning predictable responses."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock()
    return client
