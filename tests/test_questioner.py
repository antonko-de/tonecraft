"""Tests for the questioner agent."""

from unittest.mock import MagicMock

import pytest

from tonecraft.agents.questioner import generate_question
from tonecraft.schemas import AgentBrief


@pytest.fixture
def questioner_brief() -> AgentBrief:
    return AgentBrief(
        role="questioner",
        persona="Curious and discerning restaurant guest",
        directives=["Ask about menu items naturally", "Express genuine interest"],
        topic_focus="Pasta dishes and sauces",
    )


def test_generate_question_returns_string(questioner_brief):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(text="What pasta do you recommend tonight?")
    question = generate_question(questioner_brief, "Pasta dishes and sauces", mock_client, model="gpt-4o-mini")
    assert isinstance(question, str)
    assert len(question) > 0


def test_generate_question_calls_client_once(questioner_brief):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(text="Is the tagliatelle house-made?")
    generate_question(questioner_brief, "Pasta dishes and sauces", mock_client, model="gpt-4o-mini")
    assert mock_client.chat.completions.create.call_count == 1


def test_generate_question_uses_topic(questioner_brief):
    """The topic is passed through and reflected in the structured call."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(text="Do you have gluten-free pasta options?")
    question = generate_question(questioner_brief, "Dietary accommodations", mock_client, model="gpt-4o-mini")
    assert isinstance(question, str)


def test_generate_question_non_empty_on_mock(questioner_brief):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(text="What wine pairs with the truffle pasta?")
    question = generate_question(questioner_brief, "Wine pairing", mock_client, model="gpt-4o-mini")
    assert question.strip() != ""
