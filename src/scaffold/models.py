"""Core data models for Scaffold configuration and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TargetConfig(BaseModel):
    """Target LLM pipeline configuration — either command mode or direct API mode."""

    command: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_file: Path | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> "TargetConfig":
        has_command = self.command is not None
        has_direct = self.provider is not None or self.model is not None
        if has_command and has_direct:
            raise ValueError("Specify either 'command' or 'provider'+'model', not both")
        if not has_command and not has_direct:
            raise ValueError("Specify either 'command' or 'provider'+'model'")
        return self

    @property
    def is_command_mode(self) -> bool:
        return self.command is not None


class MetricThreshold(BaseModel):
    """A metric with a threshold to check against."""

    name: str
    threshold: float
    mode: Literal["absolute", "max_regression"]


class RubricCriterion(BaseModel):
    """A single criterion for LLM-as-judge evaluation."""

    id: str
    prompt: str


class JudgeConfig(BaseModel):
    """Judge configuration — determines how to score each example."""

    type: Literal["exact_match", "llm", "custom", "composite"] = "exact_match"
    model: str | None = None
    rubric: list[RubricCriterion] | str | None = None
    module: str | None = None
    function: str | None = None
    criteria: list[dict] | None = None  # Phase 6: composite judge


class DatasetSource(BaseModel):
    """Remote dataset source (v2). V1 uses plain string paths only."""

    source: str
    cache: bool = True


class EvalConfig(BaseModel):
    """Configuration for a single eval."""

    name: str
    level: Literal["prompt", "pipeline", "agent"] = "pipeline"
    dataset: str | DatasetSource
    target: TargetConfig | None = None
    judge: JudgeConfig = Field(default_factory=lambda: JudgeConfig())
    metrics: list[MetricThreshold] = Field(default_factory=list)
    mode: Literal["full_replay", "history_injection"] | None = None


class Settings(BaseModel):
    """Global settings for eval runs."""

    parallelism: int = 10
    timeout_per_call: int = 30
    retries: int = 2
    smoke_test_size: int | None = None


class ScaffoldConfig(BaseModel):
    """Root configuration model for scaffold.yaml."""

    version: int = 1
    target: TargetConfig
    evals: list[EvalConfig]
    settings: Settings = Field(default_factory=Settings)


# --- Runtime data models (not from config) ---


class EvalExample(BaseModel):
    """A single example from a JSONL eval dataset."""

    input: str
    expected: str
    extra: dict = Field(default_factory=dict)


@dataclass
class TargetResult:
    """Result from running a target on one example."""

    output: str
    latency_ms: float
    error: str | None = None


@dataclass
class JudgeResult:
    """Result from judging one example."""

    score: float  # 0.0 to 1.0
    reason: str | None = None


@dataclass
class EvalResult:
    """Full result of running one eval."""

    eval_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    per_example: list[JudgeResult] = field(default_factory=list)
    examples: list[EvalExample] = field(default_factory=list)
    results: list[TargetResult] = field(default_factory=list)
    latency_stats: dict[str, float] = field(default_factory=dict)
    num_examples: int = 0
    num_errors: int = 0
