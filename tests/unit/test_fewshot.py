"""Tests for few-shot migration strategy."""

from unittest.mock import patch

import pytest

from llmci.migrate.fewshot import build_fewshot_prompt, optimize_fewshot
from llmci.migrate.model_spec import ModelSpec
from llmci.migrate.splitter import DataSplit
from llmci.models import EvalConfig, EvalExample, JudgeConfig


def test_build_fewshot_prompt_empty():
    assert build_fewshot_prompt("Classify: {input}", []) == "Classify: {input}"


def test_build_fewshot_prompt_inlines_examples():
    ex = EvalExample(input="hello", expected="greeting")
    out = build_fewshot_prompt("Task: {input}", [ex])
    assert "## Examples" in out
    assert "hello" in out
    assert "greeting" in out
    assert "Task: {input}" in out


@pytest.mark.asyncio
async def test_optimize_fewshot_adds_example():
    split = DataSplit(
        train=[EvalExample(input="a", expected="x"), EvalExample(input="b", expected="y")],
        validation=[EvalExample(input="c", expected="x")],
        holdout=[EvalExample(input="d", expected="x")],
    )
    eval_cfg = EvalConfig(name="t", dataset="f.jsonl", judge=JudgeConfig(type="exact_match"))

    scores = {"": 0.5, "a": 0.9}

    async def mock_eval(prompt, model_spec, examples, eval_config, primary_metric):
        if "Input: a" in prompt:
            return scores["a"]
        return scores[""]

    async def mock_failures(prompt, model_spec, examples, eval_config):
        if "Input: a" not in prompt:
            return [{"input": "a", "expected": "x", "actual": "wrong", "reason": "bad"}]
        return []

    from_spec = ModelSpec.parse("openai/gpt-4o")
    to_spec = ModelSpec.parse("anthropic/claude-3-haiku-20240307")

    with (
        patch("llmci.migrate.fewshot._evaluate_prompt", side_effect=mock_eval),
        patch("llmci.migrate.fewshot._get_failures", side_effect=mock_failures),
    ):
        result = await optimize_fewshot(
            "Classify: {input}",
            from_spec,
            to_spec,
            eval_cfg,
            split,
            "accuracy",
            max_few_shot=3,
        )

    assert result.strategy == "few_shot"
    assert result.few_shot_count == 1
    assert "Input: a" in result.best_prompt
