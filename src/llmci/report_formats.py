"""Machine-readable report formats for CI portability.

The default markdown report is great for GitHub PR comments, but every other CI
system (GitLab, Bitbucket, Azure DevOps, Jenkins, CircleCI) speaks JUnit XML, and
code-scanning surfaces speak SARIF. Emitting these unlocks native test reporting
and inline annotations without any provider-specific glue.

Each formatter consumes the same ``ThresholdResult`` list the markdown report uses,
so output formats never drift from the pass/fail logic.
"""

from __future__ import annotations

import json
from xml.etree import ElementTree as ET

from llmci.baseline import Baseline
from llmci.comparison import ThresholdResult, check_thresholds
from llmci.models import EvalConfig, EvalResult

OutputFormat = str  # "markdown" | "junit" | "sarif" | "json"

VALID_FORMATS = ("markdown", "junit", "sarif", "json")


def format_report_as(
    fmt: OutputFormat,
    results: list[EvalResult],
    configs: list[EvalConfig],
    baselines: dict[str, Baseline] | None = None,
) -> tuple[str, bool]:
    """Render results in the requested format.

    Returns ``(content, all_passed)``. ``markdown`` delegates to the existing
    report so there is a single source of truth for that format.
    """
    if fmt == "markdown":
        from llmci.report import format_report

        return format_report(results, configs, baselines=baselines)

    baselines = baselines or {}
    threshold_results = check_thresholds(results, baselines, configs)
    all_passed = all(tr.passed for tr in threshold_results)

    if fmt == "junit":
        return format_junit(threshold_results, results), all_passed
    if fmt == "sarif":
        return format_sarif(threshold_results), all_passed
    if fmt == "json":
        return format_json(threshold_results, results), all_passed

    raise ValueError(
        f"Unknown output format: {fmt!r}. Valid formats: {', '.join(VALID_FORMATS)}"
    )


def _threshold_phrase(tr: ThresholdResult) -> str:
    if tr.mode == "absolute":
        return f"must be >= {tr.threshold}"
    if tr.mode == "max_regression":
        return f"drop from baseline must be <= {tr.threshold * 100:.0f}%"
    return f"threshold {tr.threshold}"


def format_junit(
    threshold_results: list[ThresholdResult],
    results: list[EvalResult],
) -> str:
    """Render results as JUnit XML.

    Each eval becomes a ``<testsuite>`` and each metric a ``<testcase>``. Failed
    thresholds emit a ``<failure>``; skipped regression checks (no baseline) emit a
    ``<skipped>``.
    """
    by_eval: dict[str, list[ThresholdResult]] = {}
    for tr in threshold_results:
        by_eval.setdefault(tr.eval_name, []).append(tr)

    errors_by_eval = {r.eval_name: r.num_errors for r in results}

    suites = ET.Element("testsuites", name="llmci")
    total_tests = total_failures = total_skipped = 0

    for eval_name, trs in by_eval.items():
        failures = sum(1 for tr in trs if not tr.passed)
        skipped = sum(
            1 for tr in trs if tr.mode == "max_regression" and tr.baseline_value is None
        )
        total_tests += len(trs)
        total_failures += failures
        total_skipped += skipped

        suite = ET.SubElement(
            suites,
            "testsuite",
            name=eval_name,
            tests=str(len(trs)),
            failures=str(failures),
            skipped=str(skipped),
            errors=str(errors_by_eval.get(eval_name, 0)),
        )
        for tr in trs:
            case = ET.SubElement(
                suite,
                "testcase",
                classname=eval_name,
                name=tr.metric_name,
            )
            is_skipped = tr.mode == "max_regression" and tr.baseline_value is None
            if is_skipped:
                ET.SubElement(case, "skipped", message="No baseline — skipped")
            elif not tr.passed:
                failure = ET.SubElement(
                    case,
                    "failure",
                    message=(
                        f"{tr.metric_name} {_threshold_phrase(tr)} "
                        f"(got {tr.current_value:.3f})"
                    ),
                )
                failure.text = tr.detail

    suites.set("tests", str(total_tests))
    suites.set("failures", str(total_failures))
    suites.set("skipped", str(total_skipped))

    xml_body = ET.tostring(suites, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body


def format_sarif(threshold_results: list[ThresholdResult]) -> str:
    """Render failed thresholds as SARIF 2.1.0.

    Only failures become SARIF ``results`` (an empty results list means "clean"
    to code-scanning consumers). Each distinct metric is registered as a rule.
    """
    rules: dict[str, dict] = {}
    sarif_results: list[dict] = []

    for tr in threshold_results:
        if tr.passed:
            continue

        rule_id = f"llmci/{tr.metric_name}"
        if rule_id not in rules:
            rules[rule_id] = {
                "id": rule_id,
                "name": tr.metric_name,
                "shortDescription": {"text": f"{tr.metric_name} regression gate"},
                "defaultConfiguration": {"level": "error"},
            }

        sarif_results.append(
            {
                "ruleId": rule_id,
                "level": "error",
                "message": {
                    "text": f"{tr.eval_name} / {tr.metric_name}: {tr.detail}"
                },
                "properties": {
                    "eval": tr.eval_name,
                    "metric": tr.metric_name,
                    "mode": tr.mode,
                    "current": tr.current_value,
                    "baseline": tr.baseline_value,
                    "threshold": tr.threshold,
                },
            }
        )

    doc = {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "llmci",
                        "informationUri": "https://github.com/llmci-cli/llmci",
                        "rules": list(rules.values()),
                    }
                },
                "results": sarif_results,
            }
        ],
    }
    return json.dumps(doc, indent=2)


def format_json(
    threshold_results: list[ThresholdResult],
    results: list[EvalResult],
) -> str:
    """Render a structured JSON summary for programmatic consumers."""
    metrics_by_eval = {r.eval_name: r.metrics for r in results}
    errors_by_eval = {r.eval_name: r.num_errors for r in results}
    examples_by_eval = {r.eval_name: r.num_examples for r in results}
    samples_by_eval = {r.eval_name: r.samples for r in results}

    evals: dict[str, dict] = {}
    for tr in threshold_results:
        bucket = evals.setdefault(
            tr.eval_name,
            {
                "name": tr.eval_name,
                "num_examples": examples_by_eval.get(tr.eval_name, 0),
                "num_errors": errors_by_eval.get(tr.eval_name, 0),
                "samples": samples_by_eval.get(tr.eval_name, 1),
                "metrics": metrics_by_eval.get(tr.eval_name, {}),
                "thresholds": [],
            },
        )
        bucket["thresholds"].append(
            {
                "metric": tr.metric_name,
                "mode": tr.mode,
                "current": tr.current_value,
                "current_ci": list(tr.current_ci) if tr.current_ci else None,
                "baseline": tr.baseline_value,
                "threshold": tr.threshold,
                "passed": tr.passed,
                "significant": tr.significant,
                "detail": tr.detail,
            }
        )

    doc = {
        "passed": all(tr.passed for tr in threshold_results),
        "evals": list(evals.values()),
    }
    return json.dumps(doc, indent=2)


__all__ = ["format_report_as", "format_junit", "format_sarif", "format_json", "VALID_FORMATS"]
