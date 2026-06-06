"""Core data models for llmci configuration and results."""

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
    base_url: str | None = None

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
    """Judge configuration — determines how to score each example.

    ``type`` is one of the built-in judges (``exact_match``, ``llm``, ``custom``,
    ``composite``, ``rag``, ``pairwise``, ``safety``) or a plugin-registered type
    (see ``llmci.plugins``). Validation of the type happens in ``create_judge`` so
    plugin types are accepted.
    """

    type: str = "exact_match"
    model: str | None = None
    rubric: list[RubricCriterion] | str | None = None
    module: str | None = None
    function: str | None = None
    criteria: list[dict] | None = None  # Phase 6: composite judge


class DatasetSource(BaseModel):
    """Remote dataset source (S3 or HTTPS)."""

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
    # Number of full eval rounds to run for flake resistance. When > 1, metrics are
    # averaged across rounds and a confidence interval is reported.
    samples_per_example: int = 1
    # Two-sided confidence level for significance gating (e.g. 0.95). When set with
    # samples_per_example > 1, a max_regression failure is only enforced if the drop
    # exceeds the threshold beyond run-to-run noise.
    significance: float | None = None


class LlmciConfig(BaseModel):
    """Root configuration model for llmci.yaml."""

    version: int = 1
    target: TargetConfig
    evals: list[EvalConfig]
    settings: Settings = Field(default_factory=Settings)
    # Dotted module paths imported at load time so their top-level register_judge()
    # calls run, enabling local/in-repo judge plugins. See llmci.plugins.
    plugins: list[str] = Field(default_factory=list)


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
    # Token usage and cost. Populated for direct API targets; command-mode targets
    # may report them via the optional "usage" / "cost" keys in their output JSON.
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    # Structured fields beyond the answer string — e.g. RAG "contexts" and
    # "retrieved_ids" from a command target's output JSON. Consumed by judges.
    metadata: dict = field(default_factory=dict)


@dataclass
class JudgeResult:
    """Result from judging one example."""

    score: float  # 0.0 to 1.0
    reason: str | None = None
    # Named per-example sub-scores (e.g. RAG faithfulness, answer_relevance). Each is
    # surfaced as a gateable aggregate metric by name (mean across examples).
    sub_scores: dict[str, float] = field(default_factory=dict)


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
    # Multi-sample (flake-resistance) fields. ``samples`` is the number of rounds;
    # ``metric_ci`` holds a (low, high) confidence interval per metric; ``significance``
    # is the confidence level used for gating (None = significance gating disabled).
    samples: int = 1
    metric_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    significance: float | None = None


# --- Agent evaluation models (v2) ---


class AgentConstraints(BaseModel):
    """Constraints on agent execution (tool budget, token budget, etc.)."""

    max_tool_calls: int | None = None
    required_tools: list[str] | None = None
    forbidden_tools: list[str] | None = None
    max_tokens: int | None = None


class AgentExpected(BaseModel):
    """Expected outcome for an agent scenario or turn."""

    outcome: str
    constraints: AgentConstraints | None = None


class AgentTurn(BaseModel):
    """A single turn in a multi-turn agent conversation."""

    user_message: str
    context: dict | None = None
    expected: AgentExpected


class AgentScenario(BaseModel):
    """Single-turn or multi-turn agent eval example."""

    input: dict | str | None = None
    expected: AgentExpected | None = None
    turns: list[AgentTurn] | None = None
    conversation_constraints: AgentConstraints | None = None

    @model_validator(mode="after")
    def validate_scenario(self) -> "AgentScenario":
        has_single = self.input is not None and self.expected is not None
        has_multi = self.turns is not None and len(self.turns) > 0
        if not has_single and not has_multi:
            raise ValueError(
                "AgentScenario requires either (input + expected) "
                "for single-turn or (turns) for multi-turn"
            )
        if has_single and has_multi:
            raise ValueError("Specify either single-turn (input+expected) or multi-turn (turns)")
        return self

    @property
    def is_multi_turn(self) -> bool:
        return self.turns is not None and len(self.turns) > 0


class TraceStep(BaseModel):
    """A single step in an agent execution trace."""

    step: int
    type: Literal["tool_call", "response"]
    tool: str | None = None
    args: dict | None = None
    content: str | None = None
    tokens: int | None = None


@dataclass
class AgentTrace:
    """Result from executing an agent scenario."""

    final_output: str | None = None
    trace: list[TraceStep] = field(default_factory=list)
    total_tool_calls: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    turns: list[dict] = field(default_factory=list)
    error: str | None = None
