"""Eval execution orchestrator."""

from __future__ import annotations

from pathlib import Path

from scaffold.dataset.loader import load_dataset
from scaffold.judges.factory import create_judge
from scaffold.metrics import compute_latency_stats, compute_metrics
from scaffold.models import (
    EvalConfig,
    EvalExample,
    EvalResult,
    JudgeResult,
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
            base_url=target.base_url,
        )


async def run_eval(
    eval_config: EvalConfig,
    target_config: TargetConfig,
    settings: Settings,
    smoke: bool = False,
    seed: int = 42,
) -> EvalResult:
    """Execute one eval end to end."""
    if eval_config.level == "agent":
        return await _run_agent_eval(eval_config, target_config, settings, smoke, seed)

    dataset_path = eval_config.dataset
    if not isinstance(dataset_path, str):
        raise ValueError("Remote dataset sources are not supported in v1")

    smoke_size = settings.smoke_test_size if smoke else None
    require_expected = eval_config.judge.type not in ("llm", "composite")
    examples = load_dataset(
        dataset_path, smoke_size=smoke_size, seed=seed,
        require_expected=require_expected,
    )

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


async def _run_agent_eval(
    eval_config: EvalConfig,
    target_config: TargetConfig,
    settings: Settings,
    smoke: bool = False,
    seed: int = 42,
) -> EvalResult:
    """Execute an agent eval end to end."""
    from scaffold.dataset.loader import load_agent_scenarios
    from scaffold.judges.composite import CompositeAgentJudge
    from scaffold.targets.agent import run_agent_target

    dataset_path = eval_config.dataset
    if not isinstance(dataset_path, str):
        raise ValueError("Remote dataset sources are not supported")

    smoke_size = settings.smoke_test_size if smoke else None
    scenarios = load_agent_scenarios(dataset_path, smoke_size=smoke_size, seed=seed)

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
