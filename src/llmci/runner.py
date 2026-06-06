"""Eval execution orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from llmci.cache import ResponseCache
from llmci.dataset.loader import load_dataset
from llmci.judges.factory import create_judge
from llmci.metrics import compute_latency_stats, compute_metrics
from llmci.models import (
    EvalConfig,
    EvalExample,
    EvalResult,
    JudgeResult,
    LlmciConfig,
    Settings,
    TargetConfig,
    TargetResult,
)

if TYPE_CHECKING:
    from llmci.baseline import Baseline


def resolve_target(eval_config: EvalConfig, global_target: TargetConfig) -> TargetConfig:
    """Per-eval target overrides global target."""
    return eval_config.target or global_target


async def run_target(
    target: TargetConfig,
    examples: list,
    settings: Settings,
    cache: ResponseCache | None = None,
) -> list[TargetResult]:
    """Dispatch to the correct target runner.

    ``cache`` is applied to direct API targets only; command-mode targets may have
    side effects and are never cached.
    """
    if target.is_command_mode:
        from llmci.targets.command import run_command_target

        return await run_command_target(
            command_template=target.command,  # type: ignore[arg-type]
            examples=examples,
            parallelism=settings.parallelism,
            timeout=settings.timeout_per_call,
            retries=settings.retries,
        )
    else:
        from llmci.targets.direct import run_direct_target

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
            base_url=target.base_url,
            cache=cache,
        )


async def run_eval(
    eval_config: EvalConfig,
    target_config: TargetConfig,
    settings: Settings,
    smoke: bool = False,
    seed: int = 42,
    cache: ResponseCache | None = None,
    baseline: "Baseline | None" = None,
) -> EvalResult:
    """Execute one eval end to end."""
    if eval_config.level == "agent":
        return await _run_agent_eval(eval_config, target_config, settings, smoke, seed)

    smoke_size = settings.smoke_test_size if smoke else None
    require_expected = eval_config.judge.type not in (
        "llm", "composite", "rag", "pairwise"
    )
    examples = load_dataset(
        eval_config.dataset, smoke_size=smoke_size, seed=seed,
        require_expected=require_expected,
    )

    target = resolve_target(eval_config, target_config)
    judge = create_judge(eval_config.judge)

    # Pairwise judging compares each output against the stored baseline output.
    from llmci.judges.pairwise import PairwiseJudge

    if isinstance(judge, PairwiseJudge):
        judge.set_baseline(baseline)

    requested_metrics = [m.name for m in eval_config.metrics]
    samples = max(1, settings.samples_per_example)

    # When sampling for flake resistance, bypass the response cache so each round is
    # an independent draw — otherwise identical cached responses would zero out the
    # variance we are trying to measure.
    round_cache = cache if samples == 1 else None

    rounds: list[tuple[list[TargetResult], list[JudgeResult]]] = []
    for _ in range(samples):
        round_results = await run_target(target, examples, settings, cache=round_cache)
        round_per_example = await judge.evaluate_dataset(examples, round_results)
        rounds.append((round_results, round_per_example))

    metrics, metric_ci = _aggregate_rounds(
        examples, rounds, requested_metrics, samples, settings.significance
    )

    # Per-example display and latency use the first round.
    first_results, first_per_example = rounds[0]
    latency_stats = compute_latency_stats(first_results)
    num_errors = sum(1 for r in first_results if r.error is not None)

    return EvalResult(
        eval_name=eval_config.name,
        metrics=metrics,
        per_example=first_per_example,
        examples=examples,
        results=first_results,
        latency_stats=latency_stats,
        num_examples=len(examples),
        num_errors=num_errors,
        samples=samples,
        metric_ci=metric_ci,
        significance=settings.significance,
    )


def _aggregate_rounds(
    examples: list[EvalExample],
    rounds: list[tuple[list[TargetResult], list[JudgeResult]]],
    requested_metrics: list[str],
    samples: int,
    significance: float | None,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Average per-round metrics and compute a confidence interval per metric."""
    from llmci.significance import confidence_interval, mean

    per_metric_values: dict[str, list[float]] = {name: [] for name in requested_metrics}
    for round_results, round_per_example in rounds:
        round_metrics = compute_metrics(
            examples, round_results, round_per_example, requested_metrics
        )
        for name in requested_metrics:
            per_metric_values[name].append(round_metrics.get(name, 0.0))

    metrics = {name: mean(values) for name, values in per_metric_values.items()}

    metric_ci: dict[str, tuple[float, float]] = {}
    if samples > 1:
        conf = significance if significance is not None else 0.95
        metric_ci = {
            name: confidence_interval(values, conf)
            for name, values in per_metric_values.items()
        }

    return metrics, metric_ci


async def _run_agent_eval(
    eval_config: EvalConfig,
    target_config: TargetConfig,
    settings: Settings,
    smoke: bool = False,
    seed: int = 42,
) -> EvalResult:
    """Execute an agent eval end to end."""
    from llmci.dataset.loader import load_agent_scenarios
    from llmci.judges.composite import CompositeAgentJudge
    from llmci.targets.agent import run_agent_target

    smoke_size = settings.smoke_test_size if smoke else None
    scenarios = load_agent_scenarios(
        eval_config.dataset, smoke_size=smoke_size, seed=seed,
    )

    target = resolve_target(eval_config, target_config)
    if not target.is_command_mode:
        raise ValueError("Agent evals require command mode target")

    mode = eval_config.mode or "full_replay"
    traces = await run_agent_target(
        command_template=target.command,  # type: ignore[arg-type]
        scenarios=scenarios,
        mode=mode,
        parallelism=settings.parallelism,
        timeout=settings.timeout_per_call,
    )

    judge = create_judge(eval_config.judge)
    if not isinstance(judge, CompositeAgentJudge):
        raise ValueError("Agent evals require a composite judge")

    per_example: list[JudgeResult] = []
    for scenario, trace in zip(scenarios, traces):
        if trace.error:
            per_example.append(JudgeResult(score=0.0, reason=f"Agent error: {trace.error}"))
        else:
            result = await judge.evaluate_scenario(scenario, trace)
            per_example.append(result)

    total_score = sum(r.score for r in per_example)
    count = len(per_example)
    computed_metrics = {
        "mean_score": total_score / count if count else 0.0,
        "pass_rate": sum(1 for r in per_example if r.score >= 0.5) / count if count else 0.0,
    }

    target_results = [
        TargetResult(
            output=t.final_output or "",
            latency_ms=t.latency_ms,
            error=t.error,
        )
        for t in traces
    ]
    examples_as_eval = [
        EvalExample(
            input=(
                s.turns[0].user_message
                if s.is_multi_turn and s.turns
                else str(s.input or "")
            ),
            expected=(
                s.expected.outcome
                if s.expected
                else (s.turns[-1].expected.outcome if s.turns else "")
            ),
        )
        for s in scenarios
    ]

    num_errors = sum(1 for t in traces if t.error is not None)
    latency_stats_data = compute_latency_stats(target_results)

    return EvalResult(
        eval_name=eval_config.name,
        metrics=computed_metrics,
        per_example=per_example,
        examples=examples_as_eval,
        results=target_results,
        latency_stats=latency_stats_data,
        num_examples=len(scenarios),
        num_errors=num_errors,
    )


async def run_all_evals(
    config: LlmciConfig,
    smoke: bool = False,
    seed: int = 42,
    cache: ResponseCache | None = None,
    baselines: "dict[str, Baseline] | None" = None,
) -> list[EvalResult]:
    """Run all evals in the config."""
    baselines = baselines or {}
    eval_results = []
    for eval_cfg in config.evals:
        result = await run_eval(
            eval_config=eval_cfg,
            target_config=config.target,
            settings=config.settings,
            smoke=smoke,
            seed=seed,
            cache=cache,
            baseline=baselines.get(eval_cfg.name),
        )
        eval_results.append(result)
    return eval_results
