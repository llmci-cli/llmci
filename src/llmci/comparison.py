"""Regression detection: compare current results against baselines."""

from __future__ import annotations

from dataclasses import dataclass

from llmci.baseline import Baseline
from llmci.models import EvalConfig, EvalResult


@dataclass
class ThresholdResult:
    eval_name: str
    metric_name: str
    baseline_value: float | None
    current_value: float
    threshold: float
    mode: str  # "absolute" or "max_regression"
    passed: bool
    detail: str


def check_thresholds(
    results: list[EvalResult],
    baselines: dict[str, Baseline],
    configs: list[EvalConfig],
) -> list[ThresholdResult]:
    """Check all metric thresholds against baselines.

    For each eval x metric:
    - absolute: current >= threshold
    - max_regression: (baseline - current) / baseline <= threshold
    """
    threshold_results: list[ThresholdResult] = []

    for result, config in zip(results, configs):
        baseline = baselines.get(result.eval_name)

        for metric in config.metrics:
            current = result.metrics.get(metric.name, 0.0)
            baseline_val = baseline.metrics.get(metric.name) if baseline else None

            passed, detail = _evaluate_threshold(
                current=current,
                baseline_val=baseline_val,
                threshold=metric.threshold,
                mode=metric.mode,
            )

            threshold_results.append(
                ThresholdResult(
                    eval_name=result.eval_name,
                    metric_name=metric.name,
                    baseline_value=baseline_val,
                    current_value=current,
                    threshold=metric.threshold,
                    mode=metric.mode,
                    passed=passed,
                    detail=detail,
                )
            )

    return threshold_results


def _evaluate_threshold(
    current: float,
    baseline_val: float | None,
    threshold: float,
    mode: str,
) -> tuple[bool, str]:
    """Evaluate a single threshold. Returns (passed, detail)."""
    if mode == "absolute":
        passed = current >= threshold
        detail = (
            f"Score {current:.3f} {'≥' if passed else '<'} threshold {threshold}"
        )
        return passed, detail

    elif mode == "max_regression":
        if baseline_val is None:
            return True, "No baseline — skipped"

        if baseline_val == 0:
            drop = 0.0
        else:
            drop = (baseline_val - current) / baseline_val

        passed = drop <= threshold
        pct = drop * 100
        threshold_pct = threshold * 100
        detail = (
            f"Dropped {pct:.1f}% ({baseline_val:.3f} → {current:.3f}, "
            f"threshold: {threshold_pct:.0f}%)"
        )
        return passed, detail

    return True, f"Unknown mode: {mode}"
