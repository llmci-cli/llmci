"""Judge factory — routes config to the correct implementation."""

from __future__ import annotations

from scaffold.errors import ConfigError
from scaffold.judges.base import Judge
from scaffold.judges.exact_match import ExactMatchJudge
from scaffold.models import JudgeConfig


def create_judge(config: JudgeConfig) -> Judge:
    """Create a judge instance from config."""
    match config.type:
        case "exact_match":
            return ExactMatchJudge()
        case "llm":
            if not config.model or not config.rubric:
                raise ConfigError("LLM judge requires 'model' and 'rubric' fields")
            raise ConfigError("LLM judge is not yet implemented (Phase 3)")
        case "custom":
            if not config.module or not config.function:
                raise ConfigError("Custom judge requires 'module' and 'function' fields")
            raise ConfigError("Custom judge is not yet implemented (Phase 3)")
        case "composite":
            raise ConfigError("Composite judges require level: agent (v2 feature)")
        case _:
            raise ConfigError(f"Unknown judge type: {config.type}")
