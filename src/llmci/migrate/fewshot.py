"""Few-shot example selection for cross-model / cross-provider migration."""

from __future__ import annotations

from dataclasses import dataclass

from llmci.migrate.model_spec import ModelSpec
from llmci.migrate.optimizer import (
    MigrationProgressEvent,
    OptimizationResult,
    OptimizationStep,
    ProgressCallback,
    _emit_progress,
    _evaluate_prompt,
    _get_failures,
    _unified_diff,
)
from llmci.migrate.splitter import DataSplit
from llmci.migrate.stopping import EarlyStopping
from llmci.models import EvalConfig, EvalExample


def build_fewshot_prompt(
    base_prompt: str, examples: list[EvalExample]
) -> str:
    """Inline selected train examples above the task prompt."""
    if not examples:
        return base_prompt
    lines = ["## Examples", ""]
    for ex in examples:
        lines.append(f"Input: {ex.input}")
        lines.append(f"Output: {ex.expected}")
        lines.append("")
    lines.extend(["## Task", "", base_prompt])
    return "\n".join(lines)


@dataclass
class FewShotCandidate:
    example: EvalExample
    val_score: float


async def optimize_fewshot(
    original_prompt: str,
    from_spec: ModelSpec,
    to_spec: ModelSpec,
    eval_config: EvalConfig,
    split: DataSplit,
    primary_metric: str,
    *,
    max_few_shot: int = 5,
    patience: int = 3,
    min_improvement: float = 0.005,
    progress_callback: ProgressCallback | None = None,
) -> OptimizationResult:
    """Greedy few-shot selection: add train examples that help the target model."""
    original_score = await _evaluate_prompt(
        original_prompt,
        from_spec,
        split.holdout,
        eval_config,
        primary_metric,
    )
    _emit_progress(
        progress_callback,
        MigrationProgressEvent(phase="baseline_complete", original_score=original_score),
    )

    selected: list[EvalExample] = []
    train_by_input = {ex.input: ex for ex in split.train}

    current_prompt = build_fewshot_prompt(original_prompt, selected)
    train_score = await _evaluate_prompt(
        current_prompt, to_spec, split.train, eval_config, primary_metric,
    )
    _emit_progress(
        progress_callback,
        MigrationProgressEvent(phase="initial_train_complete", train_score=train_score),
    )

    stopper = EarlyStopping(patience=patience, min_improvement=min_improvement)
    best_prompt = current_prompt
    best_val_score = await _evaluate_prompt(
        current_prompt, to_spec, split.validation, eval_config, primary_metric,
    )
    steps: list[OptimizationStep] = []
    stopped_reason = "max_few_shot"

    for iteration in range(1, max_few_shot + 1):
        failures = await _get_failures(
            current_prompt, to_spec, split.train, eval_config,
        )
        _emit_progress(
            progress_callback,
            MigrationProgressEvent(
                phase="iteration_start",
                iteration=iteration,
                max_iterations=max_few_shot,
                train_score=train_score,
                failure_count=len(failures),
            ),
        )

        if not failures:
            stopped_reason = "converged"
            break

        candidates: list[FewShotCandidate] = []
        for failure in failures:
            ex = train_by_input.get(failure["input"])
            if ex is None or ex in selected:
                continue
            trial = selected + [ex]
            trial_prompt = build_fewshot_prompt(original_prompt, trial)
            val_score = await _evaluate_prompt(
                trial_prompt, to_spec, split.validation, eval_config, primary_metric,
            )
            candidates.append(FewShotCandidate(example=ex, val_score=val_score))

        if not candidates:
            stopped_reason = "no_candidates"
            break

        best_candidate = max(candidates, key=lambda c: c.val_score)
        if best_candidate.val_score <= best_val_score:
            _emit_progress(
                progress_callback,
                MigrationProgressEvent(
                    phase="iteration_skipped",
                    iteration=iteration,
                    max_iterations=max_few_shot,
                    train_score=train_score,
                    failure_count=len(failures),
                    reason="no_improvement",
                ),
            )
            if stopper.should_stop(best_val_score):
                stopped_reason = "patience"
                break
            continue

        selected.append(best_candidate.example)
        current_prompt = build_fewshot_prompt(original_prompt, selected)
        train_score = await _evaluate_prompt(
            current_prompt, to_spec, split.train, eval_config, primary_metric,
        )
        best_val_score = best_candidate.val_score
        best_prompt = current_prompt

        diff = _unified_diff(
            build_fewshot_prompt(original_prompt, selected[:-1]),
            current_prompt,
        )
        steps.append(OptimizationStep(
            iteration=iteration,
            prompt_text=current_prompt,
            train_score=train_score,
            val_score=best_val_score,
            diff=diff or f"+ example: {best_candidate.example.input[:80]}",
        ))

        _emit_progress(
            progress_callback,
            MigrationProgressEvent(
                phase="iteration_complete",
                iteration=iteration,
                max_iterations=max_few_shot,
                train_score=train_score,
                val_score=best_val_score,
                failure_count=len(failures),
                accepted=True,
            ),
        )

        if stopper.should_stop(best_val_score):
            stopped_reason = "patience"
            break

    holdout_score = await _evaluate_prompt(
        best_prompt, to_spec, split.holdout, eval_config, primary_metric,
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
        from_model=from_spec.raw,
        to_model=to_spec.raw,
        steps=steps,
        stopped_reason=stopped_reason,
        strategy="few_shot",
        few_shot_count=len(selected),
    )
