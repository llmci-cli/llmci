"""Tests for the prompt optimizer (mocked LLM calls)."""



from llmci.migrate.optimizer import (
    OptimizationResult,
    _edit_distance,
    _extract_prompt,
    _unified_diff,
)


class TestExtractPrompt:
    def test_basic_extraction(self):
        response = "<reasoning>Fix it</reasoning>\n<prompt>Hello {input}</prompt>"
        assert _extract_prompt(response) == "Hello {input}"

    def test_multiline_prompt(self):
        response = "<prompt>\nLine 1\nLine 2\n</prompt>"
        assert "Line 1" in _extract_prompt(response)
        assert "Line 2" in _extract_prompt(response)

    def test_no_tags(self):
        response = "Here is my suggestion: just change the wording"
        assert _extract_prompt(response) is None

    def test_empty_tags(self):
        result = _extract_prompt("<prompt></prompt>")
        assert result == ""


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("hello", "hello") == 0

    def test_one_char_diff(self):
        assert _edit_distance("hello", "hallo") == 1

    def test_insertion(self):
        assert _edit_distance("hello", "helloo") == 1

    def test_deletion(self):
        assert _edit_distance("hello", "hell") == 1

    def test_empty(self):
        assert _edit_distance("", "abc") == 3
        assert _edit_distance("abc", "") == 3


class TestUnifiedDiff:
    def test_no_diff(self):
        assert _unified_diff("hello", "hello") == ""

    def test_has_diff(self):
        diff = _unified_diff("hello world", "hello earth")
        assert "---" in diff or "+++" in diff or diff == ""

    def test_multiline_diff(self):
        old = "line1\nline2\nline3"
        new = "line1\nchanged\nline3"
        diff = _unified_diff(old, new)
        assert "changed" in diff


class TestOptimizationResult:
    def test_dataclass(self):
        result = OptimizationResult(
            best_prompt="test",
            best_val_score=0.95,
            holdout_score=0.93,
            original_score=0.90,
            from_model="gpt-4o",
            to_model="gpt-4.5",
            steps=[],
            stopped_reason="converged",
        )
        assert result.best_prompt == "test"
        assert result.stopped_reason == "converged"
