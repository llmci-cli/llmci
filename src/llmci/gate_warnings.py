"""Warnings that help teams keep CI gates trustworthy before they silently flake."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmci.baseline import Baseline
    from llmci.models import LlmciConfig


def collect_gate_warnings(
    config: "LlmciConfig",
    baselines: dict[str, "Baseline"] | None = None,
    *,
    update_baseline: bool = False,
) -> list[str]:
    """Return advisory warnings about config choices that weaken the gate.

    These never change pass/fail — they surface misconfigurations early so a team
    doesn't ship a gate that always skips regressions or treats noise as failure.
    """
    if update_baseline:
        return []

    baselines = baselines or {}
    warnings: list[str] = []

    if config.settings.samples_per_example > 1 and config.settings.significance is None:
        warnings.append(
            "settings.samples_per_example > 1 without settings.significance — "
            "max_regression thresholds will gate on point estimates, not "
            "statistical significance. Set significance: 0.95 (or pass --significance)."
        )

    for eval_cfg in config.evals:
        name = eval_cfg.name
        has_baseline = name in baselines

        if eval_cfg.judge.type == "pairwise" and not has_baseline:
            warnings.append(
                f"eval '{name}' uses a pairwise judge but has no baseline — "
                "win_rate comparisons will score neutral (0.5) for every example."
            )

        regression_metrics = [
            m.name for m in eval_cfg.metrics if m.mode == "max_regression"
        ]
        if regression_metrics and not has_baseline:
            joined = ", ".join(regression_metrics)
            warnings.append(
                f"eval '{name}' has max_regression thresholds ({joined}) but no "
                "baseline — those checks will be skipped. Run "
                "`llmci run --update-baseline` on main, or pass --compare-to."
            )

    return warnings
