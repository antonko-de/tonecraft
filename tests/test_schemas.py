"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from tonecraft.schemas import (
    AgentBrief,
    ContextDocument,
    DistributionSchema,
    GenerationConfig,
    GenerationResult,
    ProjectConfig,
    QAPair,
    TargetConfig,
    TopicDistribution,
    TrainingExample,
)


# --- ProjectConfig ---

def test_project_config_from_dict():
    config = ProjectConfig.model_validate({
        "target": {"slm": "microsoft/phi-3-mini"},
    })
    assert config.target.slm == "microsoft/phi-3-mini"
    assert config.generation.provider == "openai"
    assert config.output.output_dir == "./output"


def test_project_config_missing_slm_raises():
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({"target": {}})


# --- ContextDocument ---

def test_context_document_valid():
    doc = ContextDocument(
        domain="Fine dining restaurant",
        questioner_role="Guest",
        responder_role="Waiter",
        tone_guidelines="Warm and professional",
        topics=["Menu items", "Wine pairing"],
        constraints=["Stay in character"],
    )
    assert doc.domain == "Fine dining restaurant"
    assert len(doc.topics) == 2


def test_context_document_missing_required_fields():
    with pytest.raises(ValidationError):
        ContextDocument(domain="test")  # missing roles, topics


def test_context_document_requires_at_least_one_topic():
    with pytest.raises(ValidationError):
        ContextDocument(
            domain="test",
            questioner_role="Q",
            responder_role="R",
            tone_guidelines="neutral",
            topics=[],
        )


# --- DistributionSchema ---

def test_distribution_schema_valid():
    schema = DistributionSchema(
        topics=[
            TopicDistribution(topic="Menu items", weight=0.6, target_count=60),
            TopicDistribution(topic="Wine pairing", weight=0.4, target_count=40),
        ],
        recommended_pair_count=100,
        data_sizing_rationale="Standard domain adaptation for a 3B model.",
    )
    assert schema.recommended_pair_count == 100
    assert len(schema.topics) == 2


def test_distribution_schema_weights_sum_to_one():
    schema = DistributionSchema(
        topics=[
            TopicDistribution(topic="A", weight=0.6, target_count=60),
            TopicDistribution(topic="B", weight=0.4, target_count=40),
        ],
        recommended_pair_count=100,
        data_sizing_rationale="test",
    )
    total = sum(t.weight for t in schema.topics)
    assert abs(total - 1.0) < 0.01


# --- QAPair ---

def test_qa_pair_valid():
    pair = QAPair(
        question="What pasta do you recommend?",
        answer="I'd suggest the tagliatelle al ragù.",
        topic="Menu items",
        confidence=0.9,
    )
    assert pair.confidence == 0.9


def test_qa_pair_confidence_too_high():
    with pytest.raises(ValidationError):
        QAPair(
            question="Q", answer="A", topic="T", confidence=1.5
        )


def test_qa_pair_confidence_too_low():
    with pytest.raises(ValidationError):
        QAPair(
            question="Q", answer="A", topic="T", confidence=-0.1
        )


# --- AgentBrief ---

def test_agent_brief_valid():
    brief = AgentBrief(
        role="waiter",
        persona="Professional and warm",
        directives=["Use 'certainly' and 'of course'", "Never break character"],
        topic_focus="Menu items",
    )
    assert brief.role == "waiter"
    assert len(brief.directives) == 2


def test_agent_brief_requires_role():
    with pytest.raises(ValidationError):
        AgentBrief(role="", persona="x", directives=["x"], topic_focus="x")


def test_agent_brief_requires_directives():
    with pytest.raises(ValidationError):
        AgentBrief(role="waiter", persona="x", directives=[], topic_focus="x")


# --- TrainingExample (chatml) ---

def test_training_example_chatml():
    example = TrainingExample(
        format="chatml",
        messages=[
            {"role": "user", "content": "What wine pairs with pasta?"},
            {"role": "assistant", "content": "I'd recommend a Barolo."},
        ],
    )
    assert example.format == "chatml"
    assert len(example.messages) == 2


def test_training_example_alpaca():
    example = TrainingExample(
        format="alpaca",
        instruction="Recommend a wine pairing.",
        input="I'm having pasta.",
        output="I'd suggest a Barolo.",
    )
    assert example.format == "alpaca"


def test_training_example_sharegpt():
    example = TrainingExample(
        format="sharegpt",
        conversations=[
            {"from": "human", "value": "What wine?"},
            {"from": "gpt", "value": "Try a Barolo."},
        ],
    )
    assert example.format == "sharegpt"


# --- GenerationResult ---

def test_generation_result_serializes_to_json():
    result = GenerationResult(
        pairs=[
            QAPair(question="Q1", answer="A1", topic="T1", confidence=0.8),
        ],
        distribution_stats={"T1": 1},
        metadata={"provider": "openai", "model": "gpt-4o-mini"},
    )
    data = result.model_dump()
    assert len(data["pairs"]) == 1
    assert data["metadata"]["provider"] == "openai"
