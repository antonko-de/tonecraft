"""Tests for the responder agent."""

from unittest.mock import MagicMock

import pytest

from tonecraft.agents.responder import generate_response
from tonecraft.schemas import AgentBrief, QAPair


@pytest.fixture
def responder_brief() -> AgentBrief:
    return AgentBrief(
        role="responder",
        persona="Professional and warm waiter with deep menu knowledge",
        directives=["Answer clearly and warmly", "Recommend with confidence"],
        topic_focus="Pasta dishes and sauces",
    )


def test_generate_response_returns_qa_pair(responder_brief):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = QAPair(
        question="What pasta do you recommend?",
        answer="The tagliatelle al ragù is excellent tonight.",
        topic="Pasta dishes and sauces",
        confidence=0.9,
    )
    pair = generate_response(responder_brief, "What pasta do you recommend?", "Pasta dishes and sauces", mock_client, model="gpt-4o-mini")
    assert isinstance(pair, QAPair)


def test_generate_response_has_answer(responder_brief):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = QAPair(
        question="What pasta do you recommend?",
        answer="The tagliatelle al ragù is excellent tonight.",
        topic="Pasta dishes and sauces",
        confidence=0.9,
    )
    pair = generate_response(responder_brief, "What pasta do you recommend?", "Pasta dishes and sauces", mock_client, model="gpt-4o-mini")
    assert pair.answer.strip() != ""


def test_generate_response_confidence_in_range(responder_brief):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = QAPair(
        question="Q",
        answer="A",
        topic="Pasta dishes and sauces",
        confidence=0.85,
    )
    pair = generate_response(responder_brief, "Q", "Pasta dishes and sauces", mock_client, model="gpt-4o-mini")
    assert 0.0 <= pair.confidence <= 1.0


def test_generate_response_preserves_question(responder_brief):
    question = "Is the pasta house-made?"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = QAPair(
        question=question,
        answer="Yes, all our pasta is made fresh daily.",
        topic="Pasta dishes and sauces",
        confidence=0.95,
    )
    pair = generate_response(responder_brief, question, "Pasta dishes and sauces", mock_client, model="gpt-4o-mini")
    assert pair.question == question


def test_generate_response_calls_client_once(responder_brief):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = QAPair(
        question="Q", answer="A", topic="T", confidence=0.8
    )
    generate_response(responder_brief, "Q", "T", mock_client, model="gpt-4o-mini")
    assert mock_client.chat.completions.create.call_count == 1
