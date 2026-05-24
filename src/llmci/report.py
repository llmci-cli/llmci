"""Markdown report generation for eval results."""

from __future__ import annotations

from llmci.baseline import Baseline
from llmci.comparison import check_thresholds
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

        threshold_str = _threshold_str(tr.threshold, tr.mode)

        if has_baselines:
            bl_str = f"{tr.baseline_value:.3f}" if tr.baseline_value is not None else "—"
            lines.append(
                f"| {tr.eval_name} | {tr.metric_name} | {bl_str} | "
                f"{tr.current_value:.3f} | {threshold_str} | {status} |"
            )
        else:
            lines.append(
                f"| {tr.eval_name} | {tr.metric_name} | "
                f"{tr.current_value:.3f} | {threshold_str} | {status} |"
            )

    lines.append("")

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

    # Error summary
    total_errors = sum(r.num_errors for r in results)
    total_examples = sum(r.num_examples for r in results)
    if total_errors > 0:
        lines.append(f"\n*{total_errors} of {total_examples} examples had target errors.*\n")

    return "\n".join(lines), all_passed


def _threshold_str(threshold: float, mode: str) -> str:
    """Human-readable threshold string."""
    if mode == "absolute":
        return f"≥ {threshold}"
    elif mode == "max_regression":
        return f"≤ {threshold * 100:.0f}% drop"
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
