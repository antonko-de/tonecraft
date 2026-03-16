"""Pydantic schemas for tonecraft. Partial — extended in A3."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v


class OutputConfig(BaseModel):
    format: list[str] = ["jsonl", "json"]
    output_dir: str = "./output"
    include_metadata: bool = True


class ProjectConfig(BaseModel):
    target: TargetConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
