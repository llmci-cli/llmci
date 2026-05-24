"""Custom user-defined judge — loads a Python function by path."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from llmci.errors import ConfigError
from llmci.judges.base import Judge
from llmci.models import JudgeResult


class CustomJudge(Judge):
    """Loads and runs a user-defined Python judge function.

    The function must accept (input: str, expected: str, actual: str)
    and return a dict with at minimum {"score": float} (0.0 to 1.0).
    Optionally includes {"score": float, "reason": str}.
    """

    def __init__(self, module_path: str, function_name: str):
        self.module_path = module_path
        self.function_name = function_name
        self.fn = self._load_function(module_path, function_name)

    def _load_function(self, module_path: str, function_name: str):
        """Dynamically import a function from a Python file."""
        path = Path(module_path)
        if not path.exists():
            raise ConfigError(
                f"Custom judge module not found: {module_path}\n\n"
                "Fix: Check the 'module' path in your judge config. "
                "It should be relative to llmci.yaml's directory."
            )

        spec = importlib.util.spec_from_file_location("custom_judge", str(path))
        if spec is None or spec.loader is None:
            raise ConfigError(f"Cannot load module: {module_path}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ConfigError(
                f"Error loading custom judge module '{module_path}': {e}"
            ) from e

        fn = getattr(module, function_name, None)
        if fn is None:
            available = [
                name for name, obj in inspect.getmembers(module)
                if callable(obj) and not name.startswith("_")
            ]
            raise ConfigError(
                f"Function '{function_name}' not found in {module_path}.\n"
                f"Available functions: {', '.join(available) or 'none'}"
            )

        return fn

    async def evaluate_single(
        self, input: str, expected: str, actual: str
    ) -> JudgeResult:
        try:
            result = self.fn(input, expected, actual)
        except Exception as e:
            return JudgeResult(
                score=0.0,
                reason=f"Custom judge error: {e}",
            )

        if not isinstance(result, dict):
            return JudgeResult(
                score=0.0,
                reason=f"Custom judge must return a dict, got {type(result).__name__}",
            )

        if "score" not in result:
            return JudgeResult(
                score=0.0,
                reason="Custom judge dict missing 'score' key",
            )

        score = float(result["score"])
        score = max(0.0, min(1.0, score))

        return JudgeResult(
            score=score,
            reason=result.get("reason"),
        )
