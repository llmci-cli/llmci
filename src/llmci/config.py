"""llmci.yaml parsing and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llmci.errors import ConfigError
from llmci.models import (
    EvalConfig,
    LlmciConfig,
    RubricCriterion,
)


def load_config(path: Path = Path("llmci.yaml")) -> LlmciConfig:
    """Load and validate llmci.yaml."""
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}\n\n"
            "Fix: Run 'llmci init' to create a llmci.yaml, "
            "or create one manually."
        )

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}:\n{e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"Expected a YAML mapping in {path}, got {type(raw).__name__}")

    raw = _normalize_config(raw)

    try:
        return LlmciConfig(**raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid config in {path}:\n{_format_validation_error(e)}") from e


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize shorthand forms before Pydantic validation."""
    if "evals" in raw:
        for eval_raw in raw["evals"]:
            if "judge" in eval_raw:
                eval_raw["judge"] = _normalize_judge(eval_raw["judge"])

            if "level" in eval_raw and eval_raw["level"] == "agent":
                if "judge" not in eval_raw or (
                    isinstance(eval_raw.get("judge"), dict)
                    and eval_raw["judge"].get("type") != "composite"
                ):
                    raise ConfigError(
                        f"Eval '{eval_raw.get('name', '?')}' uses level: agent "
                        "but does not have a composite judge. "
                        "Agent evals require judge type: composite."
                    )

    return raw


def _normalize_judge(raw: str | dict) -> dict:
    """Convert shorthand judge strings to full JudgeConfig dicts."""
    if isinstance(raw, str):
        valid_shorthands = {"exact_match", "llm", "custom"}
        if raw not in valid_shorthands:
            raise ConfigError(
                f"Unknown judge shorthand: '{raw}'. "
                f"Valid options: {', '.join(sorted(valid_shorthands))}"
            )
        return {"type": raw}

    if isinstance(raw, dict):
        if "rubric" in raw and isinstance(raw["rubric"], list):
            raw["rubric"] = [
                item if isinstance(item, RubricCriterion) else item for item in raw["rubric"]
            ]
        return raw

    raise ConfigError(f"Judge must be a string or mapping, got {type(raw).__name__}")


def _format_validation_error(e: ValidationError) -> str:
    """Format Pydantic validation errors into human-readable messages."""
    lines = []
    for err in e.errors():
        loc = " → ".join(str(part) for part in err["loc"])
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)


def find_eval(config: LlmciConfig, eval_name: str) -> EvalConfig:
    """Find an eval by name in the config."""
    for eval_cfg in config.evals:
        if eval_cfg.name == eval_name:
            return eval_cfg
    available = [e.name for e in config.evals]
    raise ConfigError(
        f"Eval '{eval_name}' not found in config. Available: {', '.join(available)}"
    )
