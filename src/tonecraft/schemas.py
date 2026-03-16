"""Pydantic schemas for tonecraft data models."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# --- Config schemas (loaded from tonecraft.toml) ---

class TargetConfig(BaseModel):
    slm: str
    slm_context_window: int = 4096
    training_format: Literal["chatml", "alpaca", "sharegpt", "custom"] = "chatml"


class GenerationConfig(BaseModel):
    provider: Literal["anthropic", "openai", "ollama"] = "openai"
    model_expert: str = "gpt-4o"
    model_agent: str = "gpt-4o-mini"
    base_url: str = ""
    max_pairs: int = 500
    max_iterations: int = 3
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class OutputConfig(BaseModel):
    format: list[str] = ["jsonl", "json"]
    output_dir: str = "./output"
    include_metadata: bool = True


class ProjectConfig(BaseModel):
    target: TargetConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# --- Domain context (parsed from .md input file) ---

class ContextDocument(BaseModel):
    domain: str
    questioner_role: str
    responder_role: str
    tone_guidelines: str
    topics: list[str] = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)

    @field_validator("questioner_role", "responder_role", "domain", "tone_guidelines")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be empty")
        return v


# --- Domain expert output ---

class TopicDistribution(BaseModel):
    topic: str
    weight: float = Field(ge=0.0, le=1.0)
    target_count: int = Field(ge=0)


class DistributionSchema(BaseModel):
    """Output from the domain expert: how to distribute QA pairs across topics."""
    topics: list[TopicDistribution]
    recommended_pair_count: int = Field(ge=1)
    data_sizing_rationale: str


# --- Agent communication ---

class AgentBrief(BaseModel):
    """Instructions passed from domain expert to questioner or responder agent."""
    role: str
    persona: str
    directives: list[str] = Field(min_length=1)
    topic_focus: str

    @field_validator("role")
    @classmethod
    def role_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("role must not be empty")
        return v


# --- Generated data ---

class QAPair(BaseModel):
    question: str
    answer: str
    topic: str
    subtopic: str = ""
    tone_label: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


# --- Training-ready output ---

class TrainingExample(BaseModel):
    """SLM-ready format. Shape depends on training_format."""
    format: Literal["chatml", "alpaca", "sharegpt", "custom"]
    # chatml
    messages: list[dict[str, str]] | None = None
    # alpaca
    instruction: str | None = None
    input: str | None = None
    output: str | None = None
    # sharegpt
    conversations: list[dict[str, str]] | None = None


# --- Final output ---

class GenerationResult(BaseModel):
    pairs: list[QAPair]
    distribution_stats: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
