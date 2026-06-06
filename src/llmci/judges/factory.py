"""Judge factory — routes config to the correct implementation."""

from __future__ import annotations

from llmci.errors import ConfigError
from llmci.judges.base import Judge
from llmci.judges.custom import CustomJudge
from llmci.judges.exact_match import ExactMatchJudge
from llmci.judges.llm_judge import LLMJudge
from llmci.models import JudgeConfig


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
            from llmci.judges.composite import CompositeAgentJudge

            return CompositeAgentJudge(
                criteria=config.criteria,
                model=config.model or "gpt-4o-mini",
            )
        case "rag":
            if not config.criteria:
                raise ConfigError("RAG judge requires 'criteria' list")
            from llmci.judges.rag import RagJudge

            try:
                return RagJudge(
                    criteria=config.criteria,
                    model=config.model or "gpt-4o-mini",
                )
            except ValueError as e:
                raise ConfigError(str(e)) from e
        case "pairwise":
            from llmci.judges.pairwise import PairwiseJudge

            criterion = config.rubric if isinstance(config.rubric, str) else None
            return PairwiseJudge(
                model=config.model or "gpt-4o-mini",
                criterion=criterion,
                position_swap=config.position_swap,
            )
        case "safety":
            if not config.criteria:
                raise ConfigError("Safety judge requires 'criteria' list")
            from llmci.judges.safety import SafetyJudge

            try:
                return SafetyJudge(
                    criteria=config.criteria,
                    model=config.model or "gpt-4o-mini",
                )
            except ValueError as e:
                raise ConfigError(str(e)) from e
        case "structured":
            from llmci.judges.structured import StructuredJudge

            schema = _load_json_schema(config.json_schema)
            try:
                return StructuredJudge(schema, partial_credit=config.partial_credit)
            except ValueError as e:
                raise ConfigError(str(e)) from e
        case _:
            return _create_plugin_judge(config)


def _load_json_schema(source: dict | str | None) -> dict:
    """Resolve the structured judge's schema from an inline dict or a .json file path."""
    if source is None:
        raise ConfigError(
            "Structured judge requires a 'json_schema' (inline object or path to a "
            ".json file)"
        )
    if isinstance(source, dict):
        return source

    from pathlib import Path

    path = Path(source)
    if not path.exists():
        raise ConfigError(f"Structured judge schema file not found: {source}")
    try:
        import json

        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in schema file {source}: {e}") from e
    if not isinstance(loaded, dict):
        raise ConfigError(f"Schema file {source} must contain a JSON object")
    return loaded


def _create_plugin_judge(config: JudgeConfig) -> Judge:
    """Build a judge from a plugin-registered type, or raise a helpful error."""
    from llmci.plugins import get_judge_factory, registered_judge_types

    factory = get_judge_factory(config.type)
    if factory is None:
        builtins = (
            "exact_match, llm, custom, composite, rag, pairwise, safety, structured"
        )
        plugins = registered_judge_types()
        plugin_hint = f"; plugins: {', '.join(plugins)}" if plugins else ""
        raise ConfigError(
            f"Unknown judge type: {config.type!r}. "
            f"Built-in types: {builtins}{plugin_hint}."
        )
    try:
        return factory(config)
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(
            f"Plugin judge {config.type!r} failed to build: {e}"
        ) from e
