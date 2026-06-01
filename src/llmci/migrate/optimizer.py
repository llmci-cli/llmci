"""Prompt optimization loop for model migration."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable

import litellm

from llmci.judges.factory import create_judge
from llmci.metrics import compute_metrics
from llmci.migrate.splitter import DataSplit
from llmci.migrate.stopping import EarlyStopping
from llmci.models import (
    EvalConfig,
    EvalExample,
)

OPTIMIZER_SYSTEM_PROMPT = """\
You are a prompt optimization assistant. You will be given:
1. The current prompt
2. The model it will run on
3. Examples where the prompt failed (input, expected output, actual output)
4. The current score and target score

Your job is to suggest a MINIMAL modification to the prompt that fixes the failures.

Rules:
- Change as little as possible. Do not rewrite from scratch.
- Prefer rewording existing instructions over adding new ones.
- Explain your reasoning in <reasoning> tags, then output the full \
modified prompt in <prompt> tags."""


@dataclass
class MigrationProgressEvent:
    """Structured progress update from the migration optimizer."""

    phase: str
    iteration: int | None = None
    max_iterations: int | None = None
    original_score: float | None = None
    train_score: float | None = None
    val_score: float | None = None
    holdout_score: float | None = None
    failure_count: int | None = None
    accepted: bool | None = None
    reason: str | None = None


ProgressCallback = Callable[[MigrationProgressEvent], None]


@dataclass
class OptimizationStep:
    iteration: int
    prompt_text: str
    train_score: float
    val_score: float
    diff: str


@dataclass
class OptimizationResult:
    best_prompt: str
    best_val_score: float
    holdout_score: float
    original_score: float
    from_model: str
    to_model: str
    steps: list[OptimizationStep] = field(default_factory=list)
    stopped_reason: str = "max_iterations"


async def optimize_prompt(
    original_prompt: str,
    from_model: str,
    to_model: str,
    optimizer_model: str,
    eval_config: EvalConfig,
    split: DataSplit,
    primary_metric: str,
    patience: int = 3,
    min_improvement: float = 0.005,
    max_iterations: int = 20,
    max_edit_distance: int | None = None,
    base_url: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> OptimizationResult:
    """Run the optimization loop to adapt a prompt from one model to another."""
    original_score = await _evaluate_prompt(
        original_prompt, from_model, split.holdout, eval_config, primary_metric,
        base_url=base_url,
    )
    _emit_progress(
        progress_callback,
        MigrationProgressEvent(phase="baseline_complete", original_score=original_score),
    )

    current_prompt = original_prompt
    train_score = await _evaluate_prompt(
        current_prompt, to_model, split.train, eval_config, primary_metric,
        base_url=base_url,
    )
    _emit_progress(
        progress_callback,
        MigrationProgressEvent(phase="initial_train_complete", train_score=train_score),
    )

    stopper = EarlyStopping(patience=patience, min_improvement=min_improvement)
    best_prompt = current_prompt
    best_val_score = 0.0
    steps: list[OptimizationStep] = []
    failures: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        failures = await _get_failures(
            current_prompt, to_model, split.train, eval_config,
            base_url=base_url,
        )
        _emit_progress(
            progress_callback,
            MigrationProgressEvent(
                phase="iteration_start",
                iteration=iteration,
                max_iterations=max_iterations,
                train_score=train_score,
                failure_count=len(failures),
            ),
        )

        if not failures:
            _emit_progress(
                progress_callback,
                MigrationProgressEvent(
                    phase="iteration_skipped",
                    iteration=iteration,
                    max_iterations=max_iterations,
                    train_score=train_score,
                    failure_count=0,
                    reason="converged",
                ),
            )
            break

        new_prompt = await _suggest_modification(
            optimizer_model=optimizer_model,
            current_prompt=current_prompt,
            to_model=to_model,
            failures=failures[:10],
            current_score=train_score,
            target_score=original_score,
        )

        if not new_prompt or new_prompt == current_prompt:
            _emit_progress(
                progress_callback,
                MigrationProgressEvent(
                    phase="iteration_skipped",
                    iteration=iteration,
                    max_iterations=max_iterations,
                    train_score=train_score,
                    failure_count=len(failures),
                    reason="no_prompt_change",
                ),
            )
            continue

        if max_edit_distance is not None:
            dist = _edit_distance(current_prompt, new_prompt)
            if dist > max_edit_distance:
                _emit_progress(
                    progress_callback,
                    MigrationProgressEvent(
                        phase="iteration_skipped",
                        iteration=iteration,
                        max_iterations=max_iterations,
                        train_score=train_score,
                        failure_count=len(failures),
                        reason="max_edit_distance",
                    ),
                )
                continue

        new_train_score = await _evaluate_prompt(
            new_prompt, to_model, split.train, eval_config, primary_metric,
            base_url=base_url,
        )
        new_val_score = await _evaluate_prompt(
            new_prompt, to_model, split.validation, eval_config, primary_metric,
            base_url=base_url,
        )

        diff = _unified_diff(current_prompt, new_prompt)
        steps.append(OptimizationStep(
            iteration=iteration,
            prompt_text=new_prompt,
            train_score=new_train_score,
            val_score=new_val_score,
            diff=diff,
        ))

        accepted = new_val_score > best_val_score
        if new_val_score > best_val_score:
            best_val_score = new_val_score
            best_prompt = new_prompt

        _emit_progress(
            progress_callback,
            MigrationProgressEvent(
                phase="iteration_complete",
                iteration=iteration,
                max_iterations=max_iterations,
                train_score=new_train_score,
                val_score=new_val_score,
                failure_count=len(failures),
                accepted=accepted,
            ),
        )

        current_prompt = new_prompt
        train_score = new_train_score

        if stopper.should_stop(new_val_score):
            stopped_reason = "patience"
            break
    else:
        stopped_reason = "max_iterations"

    if not failures:
        stopped_reason = "converged"

    holdout_score = await _evaluate_prompt(
        best_prompt, to_model, split.holdout, eval_config, primary_metric,
        base_url=base_url,
    )
    _emit_progress(
        progress_callback,
        MigrationProgressEvent(phase="complete", holdout_score=holdout_score),
    )

    return OptimizationResult(
        best_prompt=best_prompt,
        best_val_score=best_val_score,
        holdout_score=holdout_score,
        original_score=original_score,
        from_model=from_model,
        to_model=to_model,
        steps=steps,
        stopped_reason=stopped_reason,
    )


def _emit_progress(
    progress_callback: ProgressCallback | None,
    event: MigrationProgressEvent,
) -> None:
    if progress_callback is not None:
        progress_callback(event)


async def _evaluate_prompt(
    prompt: str,
    model: str,
    examples: list[EvalExample],
    eval_config: EvalConfig,
    primary_metric: str,
    base_url: str | None = None,
) -> float:
    """Evaluate a prompt on a set of examples using direct LLM calls."""
    from llmci.targets.direct import run_direct_target

    provider, _, model_name = model.rpartition("/")

    results = await run_direct_target(
        provider=provider,
        model=model_name if provider else model,
        prompt_template=prompt,
        examples=examples,
        parallelism=5,
        timeout=30,
        retries=1,
        base_url=base_url,
    )

    judge = create_judge(eval_config.judge)
    per_example = await judge.evaluate_dataset(examples, results)
    metrics = compute_metrics(
        examples, results, per_example, [primary_metric]
    )

    return metrics.get(primary_metric, 0.0)


async def _get_failures(
    prompt: str,
    model: str,
    examples: list[EvalExample],
    eval_config: EvalConfig,
    base_url: str | None = None,
) -> list[dict]:
    """Get examples where the prompt fails."""
    from llmci.targets.direct import run_direct_target

    provider, _, model_name = model.rpartition("/")

    results = await run_direct_target(
        provider=provider,
        model=model_name if provider else model,
        prompt_template=prompt,
        examples=examples,
        parallelism=5,
        timeout=30,
        retries=1,
        base_url=base_url,
    )

    judge = create_judge(eval_config.judge)
    per_example = await judge.evaluate_dataset(examples, results)

    failures = []
    for i, jr in enumerate(per_example):
        if jr.score < 0.5 and i < len(examples) and i < len(results):
            failures.append({
                "input": examples[i].input,
                "expected": examples[i].expected,
                "actual": results[i].output,
                "reason": jr.reason,
            })

    return failures


async def _suggest_modification(
    optimizer_model: str,
    current_prompt: str,
    to_model: str,
    failures: list[dict],
    current_score: float,
    target_score: float,
) -> str | None:
    """Ask the optimizer LLM to suggest a prompt modification."""
    failures_text = "\n".join(
        f"- Input: {f['input'][:200]}\n"
        f"  Expected: {f['expected'][:200]}\n"
        f"  Got: {f['actual'][:200]}"
        for f in failures
    )

    user_msg = (
        f"## Current Prompt\n{current_prompt}\n\n"
        f"## Target Model\n{to_model}\n\n"
        f"## Current Score: {current_score:.3f} (target: {target_score:.3f})\n\n"
        f"## Failure Examples\n{failures_text}\n\n"
        "Suggest a minimal modification to improve the prompt."
    )

    try:
        response = await litellm.acompletion(
            model=optimizer_model,
            messages=[
                {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            timeout=60,
        )
        content = response.choices[0].message.content or ""
        return _extract_prompt(content)
    except Exception:
        return None


def _extract_prompt(response: str) -> str | None:
    """Extract the prompt from <prompt>...</prompt> tags."""
    match = re.search(r"<prompt>(.*?)</prompt>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            current_row.append(
                min(
                    current_row[j] + 1,
                    previous_row[j + 1] + 1,
                    previous_row[j] + cost,
                )
            )
        previous_row = current_row

    return previous_row[-1]


def _unified_diff(old: str, new: str) -> str:
    """Generate a unified diff between two prompt strings."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile="before", tofile="after")
    return "".join(diff)
