"""Tests for the domain expert agent."""

from unittest.mock import MagicMock, patch

import pytest

from tonecraft.agents.domain_expert import analyze
from tonecraft.schemas import (
    AgentBrief,
    ContextDocument,
    DistributionSchema,
    ProjectConfig,
    TopicDistribution,
)


@pytest.fixture
def context_doc() -> ContextDocument:
    return ContextDocument(
        domain="Fine dining Italian restaurant with tasting menus.",
        questioner_role="Restaurant guest",
        responder_role="Waiter",
        tone_guidelines="Warm and professional.",
        topics=["Menu items", "Wine pairing", "Dietary accommodations"],
        constraints=["Stay in character"],
    )


@pytest.fixture
def project_config(sample_config_dict) -> ProjectConfig:
    return ProjectConfig.model_validate(sample_config_dict)


@pytest.fixture
def mock_distribution() -> DistributionSchema:
    return DistributionSchema(
        topics=[
            TopicDistribution(topic="Menu items", weight=0.4, target_count=40),
            TopicDistribution(topic="Wine pairing", weight=0.35, target_count=35),
            TopicDistribution(topic="Dietary accommodations", weight=0.25, target_count=25),
        ],
        recommended_pair_count=100,
        data_sizing_rationale="Phi-3-mini (3.8B params) fine-tuning typically needs 500-2000 examples for domain adaptation. Using 100 for this focused domain.",
    )


@pytest.fixture
def mock_questioner_brief() -> AgentBrief:
    return AgentBrief(
        role="questioner",
        persona="Curious and discerning restaurant guest",
        directives=["Ask about menu items", "Inquire about wine pairings"],
        topic_focus="Menu items",
    )


@pytest.fixture
def mock_responder_brief() -> AgentBrief:
    return AgentBrief(
        role="responder",
        persona="Professional and warm waiter",
        directives=["Describe dishes clearly", "Recommend wines confidently"],
        topic_focus="Menu items",
    )


def test_analyze_returns_tuple_of_three(
    context_doc, project_config, mock_distribution, mock_questioner_brief, mock_responder_brief
):
    """analyze() returns (DistributionSchema, AgentBrief, AgentBrief)."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        mock_distribution,
        mock_questioner_brief,
        mock_responder_brief,
    ]
    result = analyze(context_doc, project_config, mock_client)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_analyze_first_element_is_distribution_schema(
    context_doc, project_config, mock_distribution, mock_questioner_brief, mock_responder_brief
):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        mock_distribution,
        mock_questioner_brief,
        mock_responder_brief,
    ]
    distribution, _, _ = analyze(context_doc, project_config, mock_client)
    assert isinstance(distribution, DistributionSchema)


def test_analyze_distribution_topics_match_context(
    context_doc, project_config, mock_distribution, mock_questioner_brief, mock_responder_brief
):
    """Returned distribution topics correspond to the context's topics."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        mock_distribution,
        mock_questioner_brief,
        mock_responder_brief,
    ]
    distribution, _, _ = analyze(context_doc, project_config, mock_client)
    result_topics = {t.topic for t in distribution.topics}
    for topic in context_doc.topics:
        assert topic in result_topics


def test_analyze_distribution_has_recommended_pair_count(
    context_doc, project_config, mock_distribution, mock_questioner_brief, mock_responder_brief
):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        mock_distribution,
        mock_questioner_brief,
        mock_responder_brief,
    ]
    distribution, _, _ = analyze(context_doc, project_config, mock_client)
    assert distribution.recommended_pair_count >= 1


def test_analyze_distribution_has_sizing_rationale(
    context_doc, project_config, mock_distribution, mock_questioner_brief, mock_responder_brief
):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        mock_distribution,
        mock_questioner_brief,
        mock_responder_brief,
    ]
    distribution, _, _ = analyze(context_doc, project_config, mock_client)
    assert distribution.data_sizing_rationale
    assert len(distribution.data_sizing_rationale) > 10


def test_analyze_returns_two_agent_briefs(
    context_doc, project_config, mock_distribution, mock_questioner_brief, mock_responder_brief
):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        mock_distribution,
        mock_questioner_brief,
        mock_responder_brief,
    ]
    _, q_brief, r_brief = analyze(context_doc, project_config, mock_client)
    assert isinstance(q_brief, AgentBrief)
    assert isinstance(r_brief, AgentBrief)


def test_analyze_briefs_have_distinct_roles(
    context_doc, project_config, mock_distribution, mock_questioner_brief, mock_responder_brief
):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        mock_distribution,
        mock_questioner_brief,
        mock_responder_brief,
    ]
    _, q_brief, r_brief = analyze(context_doc, project_config, mock_client)
    assert q_brief.role != r_brief.role


def test_analyze_calls_client_three_times(
    context_doc, project_config, mock_distribution, mock_questioner_brief, mock_responder_brief
):
    """Domain expert makes 3 structured calls: distribution + 2 briefs."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        mock_distribution,
        mock_questioner_brief,
        mock_responder_brief,
    ]
    analyze(context_doc, project_config, mock_client)
    assert mock_client.chat.completions.create.call_count == 3


@pytest.mark.integration
def test_analyze_integration(context_doc, project_config):
    """Real API call — skipped without provider credentials."""
    from tonecraft.providers import create_client
    client = create_client(
        project_config.generation.provider,
        project_config.generation.model_expert,
    )
    distribution, q_brief, r_brief = analyze(context_doc, project_config, client)
    assert distribution.recommended_pair_count >= 1
    assert q_brief.role != r_brief.role
