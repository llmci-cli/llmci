"""Markdown report generation for eval results."""

from __future__ import annotations

from llmci.baseline import Baseline
from llmci.comparison import check_thresholds, compute_output_diffs
from llmci.models import EvalConfig, EvalResult


def format_report(
    results: list[EvalResult],
    configs: list[EvalConfig],
    baselines: dict[str, Baseline] | None = None,
) -> tuple[str, bool]:
    """Generate a markdown eval report.

    Phase 1 (no baselines): absolute thresholds only, no baseline column.
    Phase 2 (with baselines): full comparison table with regression detection.

    Returns (markdown_string, all_passed).
    """
    baselines = baselines or {}
    threshold_results = check_thresholds(results, baselines, configs)

    all_passed = all(tr.passed for tr in threshold_results)
    has_baselines = bool(baselines)

    lines: list[str] = []
    lines.append("## llmci Eval Report\n")

    # Summary table
    if has_baselines:
        lines.append(
            "| Eval | Metric | Baseline | This PR | Threshold | Status |"
        )
        lines.append("|------|--------|----------|---------|-----------|--------|")
    else:
        lines.append("| Eval | Metric | Score | Threshold | Status |")
        lines.append("|------|--------|-------|-----------|--------|")

    for tr in threshold_results:
        status = "✅" if tr.passed else "❌"
        if tr.mode == "max_regression" and tr.baseline_value is None:
            status = "⚠️"

        threshold_str = _threshold_str(tr.threshold, tr.mode, tr.metric_name)
        current_str = _current_str(tr.current_value, tr.current_ci)

        if has_baselines:
            bl_str = f"{tr.baseline_value:.3f}" if tr.baseline_value is not None else "—"
            lines.append(
                f"| {tr.eval_name} | {tr.metric_name} | {bl_str} | "
                f"{current_str} | {threshold_str} | {status} |"
            )
        else:
            lines.append(
                f"| {tr.eval_name} | {tr.metric_name} | "
                f"{current_str} | {threshold_str} | {status} |"
            )

    lines.append("")

    samples = max((r.samples for r in results), default=1)
    if samples > 1:
        lines.append(
            f"> Averaged over {samples} rounds; values show "
            f"the mean with a confidence interval `[low, high]`.\n"
        )

    # Regressions that were within run-to-run noise and therefore not enforced.
    waived = [tr for tr in threshold_results if tr.waived]
    if waived:
        lines.append("### Regressions Within Noise (not enforced)\n")
        for tr in waived:
            lines.append(f"**{tr.eval_name} / {tr.metric_name}:** {tr.detail}\n")

    # Warning for skipped regression checks
    if not has_baselines:
        has_regression = any(tr.mode == "max_regression" for tr in threshold_results)
        if has_regression:
            lines.append(
                "> **Note:** max_regression thresholds were skipped (no baseline found). "
                "Run `llmci run --update-baseline` on the main branch first.\n"
            )

    # Regression details
    failed_thresholds = [tr for tr in threshold_results if not tr.passed]
    if failed_thresholds:
        lines.append("### Regressions Detected\n")
        for tr in failed_thresholds:
            lines.append(f"**{tr.eval_name} / {tr.metric_name}:** {tr.detail}\n")

    # Failed examples
    any_failed = False
    for result in results:
        fails = _get_failed_examples(result)
        if fails:
            if not any_failed:
                lines.append("### Failed Examples\n")
                any_failed = True

            lines.append(
                f"<details>\n<summary>{result.eval_name}: "
                f"{len(fails)} failed</summary>\n"
            )
            lines.append("| Input (truncated) | Expected | Got | Reason |")
            lines.append("|-------------------|----------|-----|--------|")
            for f in fails[:20]:
                inp = _truncate(f["input"], 50)
                lines.append(
                    f"| {inp} | {f['expected']} | {f['actual']} | {f['reason'] or ''} |"
                )
            if len(fails) > 20:
                lines.append(f"| ... and {len(fails) - 20} more | | | |")
            lines.append("</details>\n")

    # Output diffs vs baseline (regressed examples only)
    if baselines:
        any_diff = False
        for result in results:
            diffs = compute_output_diffs(result, baselines.get(result.eval_name))
            if not diffs:
                continue
            if not any_diff:
                lines.append("### Output Diffs vs Baseline\n")
                any_diff = True
            lines.append(
                f"<details>\n<summary>{result.eval_name}: "
                f"{len(diffs)} regressed</summary>\n"
            )
            lines.append("| Input | Baseline output | This PR output | Score |")
            lines.append("|-------|-----------------|----------------|-------|")
            for d in diffs:
                lines.append(
                    f"| {_truncate(d.input, 40)} | {_truncate(d.baseline_output, 40)} "
                    f"| {_truncate(d.current_output, 40)} "
                    f"| {d.baseline_score:.2f} → {d.current_score:.2f} |"
                )
            lines.append("</details>\n")

    # Error summary
    total_errors = sum(r.num_errors for r in results)
    total_examples = sum(r.num_examples for r in results)
    if total_errors > 0:
        lines.append(f"\n*{total_errors} of {total_examples} examples had target errors.*\n")

    return "\n".join(lines), all_passed


def _current_str(value: float, ci: tuple[float, float] | None) -> str:
    """Format a current metric value, with a CI when sampling was used."""
    if ci is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _threshold_str(threshold: float, mode: str, metric_name: str = "") -> str:
    """Human-readable threshold string."""
    from llmci.metrics import is_lower_is_better

    lower = is_lower_is_better(metric_name)
    if mode == "absolute":
        return f"≤ {threshold}" if lower else f"≥ {threshold}"
    elif mode == "max_regression":
        word = "rise" if lower else "drop"
        return f"≤ {threshold * 100:.0f}% {word}"
    return str(threshold)


def _get_failed_examples(result: EvalResult) -> list[dict]:
    """Extract examples where the judge score < 0.5."""
    fails = []
    for i, jr in enumerate(result.per_example):
        if jr.score < 0.5 and i < len(result.examples) and i < len(result.results):
            fails.append({
                "input": result.examples[i].input,
                "expected": result.examples[i].expected,
                "actual": result.results[i].output,
                "reason": jr.reason,
            })
    return fails


def _truncate(s: str, max_len: int) -> str:
    """Truncate a string with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
