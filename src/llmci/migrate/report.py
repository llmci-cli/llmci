"""Migration report formatting."""

from __future__ import annotations

from llmci.migrate.optimizer import OptimizationResult, _unified_diff


def format_migration_report(result: OptimizationResult) -> str:
    """Format a migration result as a markdown report."""
    lines: list[str] = []

    lines.append(f"## Migration Report: {result.from_model} → {result.to_model}\n")

    lines.append("### Optimization Summary\n")
    lines.append(f"- Iterations: {len(result.steps)} (stopped: {result.stopped_reason})")

    if result.steps:
        lines.append(
            f"- Train score: {result.steps[0].train_score:.3f} → "
            f"{result.steps[-1].train_score:.3f}"
        )
        lines.append(
            f"- Validation score: {result.steps[0].val_score:.3f} → "
            f"{result.best_val_score:.3f}"
        )
    lines.append(f"- Holdout score: {result.holdout_score:.3f} "
                 f"(original on old model: {result.original_score:.3f})")

    parity = result.holdout_score >= result.original_score * 0.98
    if parity:
        lines.append("\n**✅ Parity achieved** — new prompt matches old model quality.")
    else:
        gap = (result.original_score - result.holdout_score) * 100
        lines.append(
            f"\n**⚠️ Gap remaining:** {gap:.1f}% below original. "
            "Consider more iterations or manual prompt tuning."
        )

    diff = _unified_diff(result.steps[0].prompt_text if result.steps else "", result.best_prompt)
    if diff:
        lines.append("\n### Prompt Diff\n")
        lines.append("```diff")
        lines.append(diff)
        lines.append("```")

    if result.steps:
        lines.append("\n### Iteration History\n")
        lines.append("| # | Train | Val | Change Summary |")
        lines.append("|---|-------|-----|----------------|")
        for step in result.steps:
            diff_summary = step.diff[:60].replace("\n", " ") if step.diff else "—"
            lines.append(
                f"| {step.iteration} | {step.train_score:.3f} | "
                f"{step.val_score:.3f} | {diff_summary} |"
            )

    return "\n".join(lines)
