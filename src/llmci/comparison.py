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
    current_ci: tuple[float, float] | None = None
    # True when a max_regression drop exceeds the threshold beyond run-to-run noise.
    # None when significance gating did not apply.
    significant: bool | None = None
    # True when a regression exceeded the threshold on the point estimate but was
    # within run-to-run noise, so it was reported but not enforced.
    waived: bool = False


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
            ci = result.metric_ci.get(metric.name)
            # Significance gating only applies when we have a CI and a configured level.
            significance = (
                result.significance if (ci is not None and result.significance) else None
            )

            from llmci.metrics import LOWER_IS_BETTER

            passed, detail, significant = _evaluate_threshold(
                current=current,
                baseline_val=baseline_val,
                threshold=metric.threshold,
                mode=metric.mode,
                ci=ci,
                significance=significance,
                lower_is_better=metric.name in LOWER_IS_BETTER,
            )

            waived = significant is False and "not significant" in detail

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
                    current_ci=ci,
                    significant=significant,
                    waived=waived,
                )
            )

    return threshold_results


def _evaluate_threshold(
    current: float,
    baseline_val: float | None,
    threshold: float,
    mode: str,
    ci: tuple[float, float] | None = None,
    significance: float | None = None,
    lower_is_better: bool = False,
) -> tuple[bool, str, bool | None]:
    """Evaluate a single threshold. Returns (passed, detail, significant).

    For ``lower_is_better`` metrics (cost, tokens, latency, error_rate) the
    comparison is inverted: absolute passes when ``current <= threshold`` and a
    *regression* is an increase rather than a drop.
    """
    if mode == "absolute":
        if lower_is_better:
            passed = current <= threshold
            sym = "≤" if passed else ">"
        else:
            passed = current >= threshold
            sym = "≥" if passed else "<"
        detail = f"Score {current:.3f} {sym} threshold {threshold}"
        return passed, detail, None

    elif mode == "max_regression":
        if baseline_val is None:
            return True, "No baseline — skipped", None

        # Regression fraction: for higher-is-better a drop is bad; for
        # lower-is-better an increase is bad.
        if baseline_val == 0:
            regression = 0.0
        elif lower_is_better:
            regression = (current - baseline_val) / baseline_val
        else:
            regression = (baseline_val - current) / baseline_val

        verb = "Rose" if lower_is_better else "Dropped"
        pct = regression * 100
        threshold_pct = threshold * 100
        point_detail = (
            f"{verb} {pct:.1f}% ({baseline_val:.3f} → {current:.3f}, "
            f"threshold: {threshold_pct:.0f}%)"
        )

        # Without a CI + significance level, gate on the point estimate.
        if ci is None or significance is None:
            return regression <= threshold, point_detail, None

        # With significance gating, use the optimistic end of the CI (the smallest
        # plausible regression). A regression only fails the gate if even the
        # optimistic estimate exceeds the threshold — i.e. it is real beyond noise.
        if not baseline_val:
            optimistic_regression = 0.0
        elif lower_is_better:
            optimistic_regression = (ci[0] - baseline_val) / baseline_val
        else:
            optimistic_regression = (baseline_val - ci[1]) / baseline_val
        significant = optimistic_regression > threshold
        passed = not significant

        conf_pct = significance * 100
        if regression <= threshold:
            detail = point_detail
        elif significant:
            detail = (
                f"{point_detail}; significant at {conf_pct:.0f}% "
                f"(CI [{ci[0]:.3f}, {ci[1]:.3f}])"
            )
        else:
            detail = (
                f"{point_detail}; within noise at {conf_pct:.0f}% "
                f"(CI [{ci[0]:.3f}, {ci[1]:.3f}]) — not significant"
            )
        return passed, detail, significant

    return True, f"Unknown mode: {mode}", None
