"""Integration test for the optimize_prompt loop with mocked LLM calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmci.migrate.optimizer import optimize_prompt
from llmci.migrate.splitter import DataSplit
from llmci.models import EvalConfig, EvalExample, JudgeConfig


def _make_examples(n: int, category: str = "billing") -> list[EvalExample]:
    return [EvalExample(input=f"ticket {i}", expected=category) for i in range(n)]


def _make_split() -> DataSplit:
    return DataSplit(
        train=_make_examples(8, "billing"),
        validation=_make_examples(4, "billing"),
        holdout=_make_examples(4, "billing"),
    )


def _make_eval_config() -> EvalConfig:
    return EvalConfig(
        name="test",
        dataset="fake.jsonl",
        judge=JudgeConfig(type="exact_match"),
        metrics=[],
    )


def _mock_completion(output: str):
    """Create a mock litellm completion response."""
    choice = MagicMock()
    choice.message.content = output
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
async def test_optimizer_converges_when_all_pass():
    """When all examples pass on iteration 1, optimizer should converge quickly."""
    eval_cfg = _make_eval_config()
    split = _make_split()

    with (
        patch(
            "llmci.migrate.optimizer._evaluate_prompt",
            new_callable=AsyncMock,
            return_value=1.0,
        ),
        patch(
            "llmci.migrate.optimizer._get_failures",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = await optimize_prompt(
            original_prompt="Classify: {input}",
            from_model="gpt-4o",
            to_model="gpt-4.5",
            optimizer_model="gpt-4o",
            eval_config=eval_cfg,
            split=split,
            primary_metric="accuracy",
            max_iterations=5,
        )

    assert result.stopped_reason == "converged"
    assert len(result.steps) == 0


@pytest.mark.asyncio
async def test_optimizer_stops_on_patience():
    """When improvements plateau, early stopping should kick in."""
    eval_cfg = _make_eval_config()
    split = _make_split()

    call_count = 0

    async def mock_evaluate(prompt, model, examples, config, metric, **kwargs):
        return 0.7

    async def mock_failures(prompt, model, examples, config, **kwargs):
        nonlocal call_count
        call_count += 1
        return [{"input": "x", "expected": "billing", "actual": "account", "reason": "wrong"}]

    async def mock_suggest(**kwargs):
        return f"Modified prompt v{call_count}: {{input}}"

    with (
        patch("llmci.migrate.optimizer._evaluate_prompt", side_effect=mock_evaluate),
        patch("llmci.migrate.optimizer._get_failures", side_effect=mock_failures),
        patch("llmci.migrate.optimizer._suggest_modification", side_effect=mock_suggest),
    ):
        result = await optimize_prompt(
            original_prompt="Classify: {input}",
            from_model="gpt-4o",
            to_model="gpt-4.5",
            optimizer_model="gpt-4o",
            eval_config=eval_cfg,
            split=split,
            primary_metric="accuracy",
            patience=2,
            max_iterations=10,
        )

    assert result.stopped_reason == "patience"
    assert len(result.steps) >= 2


@pytest.mark.asyncio
async def test_optimizer_respects_max_iterations():
    """Optimizer should stop at max_iterations."""
    eval_cfg = _make_eval_config()
    split = _make_split()
    iteration = 0

    async def mock_evaluate(prompt, model, examples, config, metric, **kwargs):
        nonlocal iteration
        iteration += 1
        return 0.5 + iteration * 0.02

    async def mock_failures(prompt, model, examples, config, **kwargs):
        return [{"input": "x", "expected": "y", "actual": "z", "reason": "wrong"}]

    async def mock_suggest(**kwargs):
        return f"Prompt v{iteration}: {{input}}"

    with (
        patch("llmci.migrate.optimizer._evaluate_prompt", side_effect=mock_evaluate),
        patch("llmci.migrate.optimizer._get_failures", side_effect=mock_failures),
        patch("llmci.migrate.optimizer._suggest_modification", side_effect=mock_suggest),
    ):
        result = await optimize_prompt(
            original_prompt="Classify: {input}",
            from_model="gpt-4o",
            to_model="gpt-4.5",
            optimizer_model="gpt-4o",
            eval_config=eval_cfg,
            split=split,
            primary_metric="accuracy",
            patience=100,
            max_iterations=3,
        )

    assert result.stopped_reason == "max_iterations"
    assert len(result.steps) == 3


@pytest.mark.asyncio
async def test_optimizer_emits_progress_events():
    """Progress callback should receive live updates for each optimization phase."""
    eval_cfg = _make_eval_config()
    split = _make_split()
    events = []

    async def mock_evaluate(prompt, model_spec, examples, config, metric):
        if model_spec.model == "gpt-4o":
            return 0.9
        if prompt.startswith("Modified"):
            return 0.8 if len(examples) == len(split.train) else 0.75
        return 0.5

    async def mock_failures(prompt, model_spec, examples, config):
        return [{"input": "x", "expected": "billing", "actual": "account", "reason": "wrong"}]

    async def mock_suggest(**kwargs):
        return "Modified prompt: {input}"

    with (
        patch("llmci.migrate.optimizer._evaluate_prompt", side_effect=mock_evaluate),
        patch("llmci.migrate.optimizer._get_failures", side_effect=mock_failures),
        patch("llmci.migrate.optimizer._suggest_modification", side_effect=mock_suggest),
    ):
        result = await optimize_prompt(
            original_prompt="Classify: {input}",
            from_model="gpt-4o",
            to_model="gpt-4.5",
            optimizer_model="gpt-4o",
            eval_config=eval_cfg,
            split=split,
            primary_metric="accuracy",
            patience=10,
            max_iterations=1,
            progress_callback=events.append,
        )

    assert result.steps
    assert [event.phase for event in events] == [
        "baseline_complete",
        "initial_train_complete",
        "iteration_start",
        "iteration_complete",
        "complete",
    ]
    assert events[0].original_score == 0.9
    assert events[1].train_score == 0.5
    assert events[2].iteration == 1
    assert events[2].failure_count == 1
    assert events[3].train_score == 0.8
    assert events[3].val_score == 0.75
    assert events[3].accepted is True
    assert events[-1].holdout_score == 0.75


@pytest.mark.asyncio
async def test_optimizer_emits_converged_skip_event():
    """When no failures remain, progress should explain why the loop stopped."""
    eval_cfg = _make_eval_config()
    split = _make_split()
    events = []

    with (
        patch(
            "llmci.migrate.optimizer._evaluate_prompt",
            new_callable=AsyncMock,
            return_value=1.0,
        ),
        patch(
            "llmci.migrate.optimizer._get_failures",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = await optimize_prompt(
            original_prompt="Classify: {input}",
            from_model="gpt-4o",
            to_model="gpt-4.5",
            optimizer_model="gpt-4o",
            eval_config=eval_cfg,
            split=split,
            primary_metric="accuracy",
            max_iterations=3,
            progress_callback=events.append,
        )

    assert result.stopped_reason == "converged"
    assert any(
        event.phase == "iteration_skipped" and event.reason == "converged"
        for event in events
    )
