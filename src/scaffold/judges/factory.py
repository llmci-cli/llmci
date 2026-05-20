"""Judge factory — routes config to the correct implementation."""

from __future__ import annotations

from scaffold.errors import ConfigError
from scaffold.judges.base import Judge
from scaffold.judges.custom import CustomJudge
from scaffold.judges.exact_match import ExactMatchJudge
from scaffold.judges.llm_judge import LLMJudge
from scaffold.models import JudgeConfig


def create_judge(config: JudgeConfig) -> Judge:
    """Create a judge instance from config."""
    match config.type:
        case "exact_match":
            return ExactMatchJudge()
        case "llm":
            if not config.model or not config.rubric:
                raise ConfigError("LLM judge requires 'model' and 'rubric' fields")
            rubric: list[dict] | str
            if isinstance(config.rubric, list):
                rubric = [{"id": r.id, "prompt": r.prompt} for r in config.rubric]
            else:
                rubric = config.rubric
            return LLMJudge(model=config.model, rubric=rubric)
        case "custom":
            if not config.module or not config.function:
                raise ConfigError("Custom judge requires 'module' and 'function' fields")
            return CustomJudge(
                module_path=config.module, function_name=config.function
            )
        case "composite":
            if not config.criteria:
                raise ConfigError("Composite judge requires 'criteria' list")
            from scaffold.judges.composite import CompositeAgentJudge

            return CompositeAgentJudge(
                criteria=config.criteria,
                model=config.model or "gpt-4o-mini",
            )
        case _:
            raise ConfigError(f"Unknown judge type: {config.type}")
