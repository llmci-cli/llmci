"""Eval execution orchestrator."""

from __future__ import annotations

from pathlib import Path

from scaffold.dataset.loader import load_dataset
from scaffold.judges.factory import create_judge
from scaffold.metrics import compute_latency_stats, compute_metrics
from scaffold.models import (
    EvalConfig,
    EvalResult,
    ScaffoldConfig,
    Settings,
    TargetConfig,
    TargetResult,
)


def resolve_target(eval_config: EvalConfig, global_target: TargetConfig) -> TargetConfig:
    """Per-eval target overrides global target."""
    return eval_config.target or global_target


async def run_target(
    target: TargetConfig,
    examples: list,
    settings: Settings,
) -> list[TargetResult]:
    """Dispatch to the correct target runner."""
    if target.is_command_mode:
        from scaffold.targets.command import run_command_target

        return await run_command_target(
            command_template=target.command,  # type: ignore[arg-type]
            examples=examples,
            parallelism=settings.parallelism,
            timeout=settings.timeout_per_call,
            retries=settings.retries,
        )
    else:
        from scaffold.targets.direct import run_direct_target

        prompt_template = ""
        if target.prompt_file:
            prompt_template = Path(target.prompt_file).read_text()
        else:
            prompt_template = "{input}"

        return await run_direct_target(
            provider=target.provider or "",
            model=target.model or "",
            prompt_template=prompt_template,
            examples=examples,
            parallelism=settings.parallelism,
            timeout=settings.timeout_per_call,
            retries=settings.retries,
        )


async def run_eval(
    eval_config: EvalConfig,
    target_config: TargetConfig,
    settings: Settings,
    smoke: bool = False,
    seed: int = 42,
) -> EvalResult:
    """Execute one eval end to end."""
    dataset_path = eval_config.dataset
    if not isinstance(dataset_path, str):
        raise ValueError("Remote dataset sources are not supported in v1")

    smoke_size = settings.smoke_test_size if smoke else None
    examples = load_dataset(dataset_path, smoke_size=smoke_size, seed=seed)

    target = resolve_target(eval_config, target_config)
    results = await run_target(target, examples, settings)

    judge = create_judge(eval_config.judge)
    per_example = await judge.evaluate_dataset(examples, results)

    requested_metrics = [m.name for m in eval_config.metrics]
    metrics = compute_metrics(examples, results, per_example, requested_metrics)
    latency_stats = compute_latency_stats(results)

    num_errors = sum(1 for r in results if r.error is not None)

    return EvalResult(
        eval_name=eval_config.name,
        metrics=metrics,
        per_example=per_example,
        examples=examples,
        results=results,
        latency_stats=latency_stats,
        num_examples=len(examples),
        num_errors=num_errors,
    )


async def run_all_evals(
    config: ScaffoldConfig,
    smoke: bool = False,
    seed: int = 42,
) -> list[EvalResult]:
    """Run all evals in the config."""
    eval_results = []
    for eval_cfg in config.evals:
        result = await run_eval(
            eval_config=eval_cfg,
            target_config=config.target,
            settings=config.settings,
            smoke=smoke,
            seed=seed,
        )
        eval_results.append(result)
    return eval_results
