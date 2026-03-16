"""Tests for the multi-provider LLM client factory."""

from unittest.mock import MagicMock, patch

import pytest

from tonecraft.providers import create_client


def test_create_openai_client():
    with patch("tonecraft.providers.openai") as mock_openai, \
         patch("tonecraft.providers.instructor") as mock_instructor:
        mock_openai.OpenAI.return_value = MagicMock()
        mock_instructor.from_openai.return_value = MagicMock()
        client = create_client("openai", model="gpt-4o")
        mock_instructor.from_openai.assert_called_once()
        assert client is not None


def test_create_anthropic_client():
    with patch("tonecraft.providers.anthropic") as mock_anthropic, \
         patch("tonecraft.providers.instructor") as mock_instructor:
        mock_anthropic.Anthropic.return_value = MagicMock()
        mock_instructor.from_anthropic.return_value = MagicMock()
        client = create_client("anthropic", model="claude-sonnet-4-20250514")
        mock_instructor.from_anthropic.assert_called_once()
        assert client is not None


def test_create_ollama_client():
    with patch("tonecraft.providers.openai") as mock_openai, \
         patch("tonecraft.providers.instructor") as mock_instructor:
        mock_openai.OpenAI.return_value = MagicMock()
        mock_instructor.from_openai.return_value = MagicMock()
        client = create_client("ollama", model="llama3", base_url="http://localhost:11434/v1")
        mock_instructor.from_openai.assert_called_once()
        assert client is not None


def test_create_ollama_uses_default_base_url():
    with patch("tonecraft.providers.openai") as mock_openai, \
         patch("tonecraft.providers.instructor") as mock_instructor:
        mock_openai.OpenAI.return_value = MagicMock()
        mock_instructor.from_openai.return_value = MagicMock()
        create_client("ollama", model="llama3")
        call_kwargs = str(mock_openai.OpenAI.call_args)
        assert "localhost:11434" in call_kwargs


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="anthropic.*openai.*ollama"):
        create_client("bedrock", model="some-model")


def test_missing_anthropic_package_raises_import_error():
    with patch("tonecraft.providers.anthropic", None):
        with pytest.raises(ImportError, match="tonecraft\\[anthropic\\]"):
            create_client("anthropic", model="claude-sonnet-4-20250514")


def test_missing_openai_package_raises_import_error():
    with patch("tonecraft.providers.openai", None):
        with pytest.raises(ImportError, match="tonecraft\\[openai\\]"):
            create_client("openai", model="gpt-4o")
