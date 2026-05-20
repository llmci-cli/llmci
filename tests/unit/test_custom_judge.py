"""Tests for custom user-defined judge."""

import pytest

from scaffold.errors import ConfigError
from scaffold.judges.custom import CustomJudge


@pytest.fixture
def write_judge(tmp_path):
    """Helper to write a custom judge Python file."""
    def _write(code: str, name: str = "my_judge.py") -> str:
        p = tmp_path / name
        p.write_text(code)
        return str(p)
    return _write


class TestCustomJudge:
    def test_load_valid_function(self, write_judge):
        path = write_judge("""
def evaluate(input, expected, actual):
    return {"score": 1.0 if actual == expected else 0.0}
""")
        judge = CustomJudge(module_path=path, function_name="evaluate")
        assert judge.fn is not None

    def test_missing_module(self):
        with pytest.raises(ConfigError, match="not found"):
            CustomJudge(module_path="/nonexistent/judge.py", function_name="fn")

    def test_missing_function(self, write_judge):
        path = write_judge("def other_fn(): pass")
        with pytest.raises(ConfigError, match="not found"):
            CustomJudge(module_path=path, function_name="evaluate")

    def test_available_functions_in_error(self, write_judge):
        path = write_judge("def my_func(): pass\ndef another(): pass")
        with pytest.raises(ConfigError, match="my_func"):
            CustomJudge(module_path=path, function_name="evaluate")

    @pytest.mark.asyncio
    async def test_evaluate_match(self, write_judge):
        path = write_judge("""
def evaluate(input, expected, actual):
    return {"score": 1.0 if actual == expected else 0.0}
""")
        judge = CustomJudge(module_path=path, function_name="evaluate")
        result = await judge.evaluate_single("q", "hello", "hello")
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_mismatch(self, write_judge):
        path = write_judge("""
def evaluate(input, expected, actual):
    return {"score": 0.0, "reason": "Not matching"}
""")
        judge = CustomJudge(module_path=path, function_name="evaluate")
        result = await judge.evaluate_single("q", "hello", "world")
        assert result.score == 0.0
        assert result.reason == "Not matching"

    @pytest.mark.asyncio
    async def test_runtime_error_handled(self, write_judge):
        path = write_judge("""
def evaluate(input, expected, actual):
    raise ValueError("Something went wrong")
""")
        judge = CustomJudge(module_path=path, function_name="evaluate")
        result = await judge.evaluate_single("q", "a", "b")
        assert result.score == 0.0
        assert "error" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_bad_return_type(self, write_judge):
        path = write_judge("""
def evaluate(input, expected, actual):
    return "not a dict"
""")
        judge = CustomJudge(module_path=path, function_name="evaluate")
        result = await judge.evaluate_single("q", "a", "b")
        assert result.score == 0.0
        assert "dict" in result.reason

    @pytest.mark.asyncio
    async def test_missing_score_key(self, write_judge):
        path = write_judge("""
def evaluate(input, expected, actual):
    return {"result": True}
""")
        judge = CustomJudge(module_path=path, function_name="evaluate")
        result = await judge.evaluate_single("q", "a", "b")
        assert result.score == 0.0
        assert "score" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_score_clamped(self, write_judge):
        path = write_judge("""
def evaluate(input, expected, actual):
    return {"score": 5.0}
""")
        judge = CustomJudge(module_path=path, function_name="evaluate")
        result = await judge.evaluate_single("q", "a", "b")
        assert result.score == 1.0

    def test_import_error_handled(self, write_judge):
        path = write_judge("import nonexistent_module_xyz")
        with pytest.raises(ConfigError, match="Error loading"):
            CustomJudge(module_path=path, function_name="evaluate")
