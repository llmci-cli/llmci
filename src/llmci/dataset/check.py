"""Dataset coverage analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from llmci.dataset.loader import load_dataset
from llmci.models import EvalExample

DEFAULT_MIN_PER_CATEGORY = 5


@dataclass
class CoverageReport:
    total_examples: int
    categories: dict[str, int]
    duplicates: list[str]
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    length_stats: dict[str, float] = field(default_factory=dict)


def check_dataset(
    path: str | Path,
    min_per_category: int = DEFAULT_MIN_PER_CATEGORY,
) -> CoverageReport:
    """Analyze dataset for coverage gaps, duplicates, and outliers."""
    examples = load_dataset(path)

    categories = Counter(ex.expected.strip() for ex in examples)
    duplicates = _find_duplicates(examples)
    length_stats = _compute_length_stats(examples)

    warnings: list[str] = []
    suggestions: list[str] = []

    if len(examples) < 10:
        warnings.append(f"Dataset is very small ({len(examples)} examples)")
        suggestions.append("Aim for at least 20-50 examples for reliable eval results")

    for cat, count in sorted(categories.items()):
        if count < min_per_category:
            warnings.append(
                f"Category '{cat}' is underrepresented ({count} examples)"
            )
            suggestions.append(
                f"Add {min_per_category - count} more examples for '{cat}'"
            )

    if len(categories) == 1:
        warnings.append(
            "All examples have the same expected value "
            "— not useful for classification metrics"
        )

    if duplicates:
        warnings.append(f"Found {len(duplicates)} duplicate input(s)")
        suggestions.append("Remove or deduplicate repeated inputs")

    outliers = _find_length_outliers(examples, length_stats)
    if outliers:
        warnings.append(f"{len(outliers)} input(s) have unusual length (>2 std devs from mean)")

    cat_imbalance = _check_imbalance(categories)
    if cat_imbalance:
        warnings.append(cat_imbalance)

    return CoverageReport(
        total_examples=len(examples),
        categories=dict(categories),
        duplicates=duplicates,
        warnings=warnings,
        suggestions=suggestions,
        length_stats=length_stats,
    )


def format_coverage_report(report: CoverageReport) -> str:
    """Format a coverage report as human-readable text."""
    lines: list[str] = []
    lines.append(f"Dataset: {report.total_examples} examples\n")

    lines.append("Categories:")
    for cat, count in sorted(report.categories.items(), key=lambda x: -x[1]):
        pct = count / report.total_examples * 100 if report.total_examples else 0
        bar = "█" * int(pct / 5)
        lines.append(f"  {cat:20s} {count:4d} ({pct:5.1f}%) {bar}")

    if report.length_stats:
        lines.append(
            f"\nInput lengths: "
            f"mean={report.length_stats.get('mean', 0):.0f}, "
            f"min={report.length_stats.get('min', 0):.0f}, "
            f"max={report.length_stats.get('max', 0):.0f}"
        )

    if report.warnings:
        lines.append("\nWarnings:")
        for w in report.warnings:
            lines.append(f"  ⚠  {w}")

    if report.suggestions:
        lines.append("\nSuggestions:")
        for s in report.suggestions:
            lines.append(f"  →  {s}")

    if not report.warnings:
        lines.append("\n✅ Dataset looks healthy!")

    return "\n".join(lines)


def _find_duplicates(examples: list[EvalExample]) -> list[str]:
    """Find duplicate input strings."""
    counts = Counter(ex.input.strip() for ex in examples)
    return [inp for inp, count in counts.items() if count > 1]


def _compute_length_stats(examples: list[EvalExample]) -> dict[str, float]:
    """Compute input length statistics."""
    if not examples:
        return {}

    lengths = [len(ex.input) for ex in examples]
    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((x - mean) ** 2 for x in lengths) / n if n > 1 else 0
    std = variance**0.5

    return {
        "mean": mean,
        "std": std,
        "min": float(min(lengths)),
        "max": float(max(lengths)),
    }


def _find_length_outliers(
    examples: list[EvalExample], stats: dict[str, float]
) -> list[int]:
    """Find indices of examples with input length >2 std devs from mean."""
    if not stats or stats.get("std", 0) == 0:
        return []

    mean = stats["mean"]
    std = stats["std"]
    threshold = 2 * std

    return [
        i for i, ex in enumerate(examples) if abs(len(ex.input) - mean) > threshold
    ]


def _check_imbalance(categories: Counter) -> str | None:
    """Check for severe class imbalance."""
    if len(categories) < 2:
        return None

    counts = list(categories.values())
    ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")

    if ratio > 5:
        largest = max(categories, key=lambda k: categories[k])
        smallest = min(categories, key=lambda k: categories[k])
        return (
            f"Severe class imbalance: '{largest}' has {categories[largest]}x "
            f"more examples than '{smallest}' ({categories[smallest]})"
        )
    return None
