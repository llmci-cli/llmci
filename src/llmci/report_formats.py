"""Machine-readable and shareable report formats.

The default markdown report is great for GitHub PR comments, but every other CI
system (GitLab, Bitbucket, Azure DevOps, Jenkins, CircleCI) speaks JUnit XML, and
code-scanning surfaces speak SARIF. ``html`` produces a self-contained, shareable
run report (upload it as a CI artifact). Emitting these unlocks native test
reporting and inline annotations without any provider-specific glue.

Each formatter consumes the same ``ThresholdResult`` list the markdown report uses,
so output formats never drift from the pass/fail logic.
"""

from __future__ import annotations

import html
import json
from xml.etree import ElementTree as ET

from llmci.baseline import Baseline
from llmci.comparison import ThresholdResult, check_thresholds, compute_output_diffs
from llmci.models import EvalConfig, EvalResult

OutputFormat = str  # "markdown" | "junit" | "sarif" | "json" | "html"

VALID_FORMATS = ("markdown", "junit", "sarif", "json", "html")


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
    if fmt == "html":
        return format_html(threshold_results, results, baselines), all_passed

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


_HTML_STYLE = """\
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 2rem auto; max-width: 60rem; padding: 0 1rem; line-height: 1.5; }
h1 { margin-bottom: 0.25rem; }
.banner { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 0.5rem;
          font-weight: 600; margin: 0.5rem 0 1.5rem; }
.banner.pass { background: #e6f4ea; color: #137333; }
.banner.fail { background: #fce8e6; color: #c5221f; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; font-size: 0.9rem; }
th { background: #f5f5f5; }
.status-pass { color: #137333; font-weight: 600; }
.status-fail { color: #c5221f; font-weight: 600; }
.status-warn { color: #b06000; font-weight: 600; }
.muted { color: #666; font-size: 0.85rem; }
details { margin: 0.75rem 0; }
summary { cursor: pointer; font-weight: 600; }
code { background: rgba(127,127,127,0.15); padding: 0 0.25rem; border-radius: 0.25rem; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _status_cell(tr: ThresholdResult) -> str:
    if tr.mode == "max_regression" and tr.baseline_value is None:
        return '<span class="status-warn">skipped</span>'
    if tr.passed:
        return '<span class="status-pass">pass</span>'
    return '<span class="status-fail">fail</span>'


def format_html(
    threshold_results: list[ThresholdResult],
    results: list[EvalResult],
    baselines: dict[str, Baseline] | None = None,
) -> str:
    """Render a self-contained HTML run report suitable as a CI artifact."""
    baselines = baselines or {}
    all_passed = all(tr.passed for tr in threshold_results)
    has_baselines = any(tr.baseline_value is not None for tr in threshold_results)
    by_eval: dict[str, list[ThresholdResult]] = {}
    for tr in threshold_results:
        by_eval.setdefault(tr.eval_name, []).append(tr)
    results_by_name = {r.eval_name: r for r in results}

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append("<title>llmci Eval Report</title>")
    parts.append(f"<style>{_HTML_STYLE}</style></head><body>")
    parts.append("<h1>llmci Eval Report</h1>")
    banner = "pass" if all_passed else "fail"
    label = "All thresholds passed" if all_passed else "Regression detected"
    parts.append(f'<div class="banner {banner}">{label}</div>')

    # Summary table
    parts.append("<h2>Summary</h2>")
    if has_baselines:
        header = ["Eval", "Metric", "Baseline", "Current", "Threshold", "Status"]
    else:
        header = ["Eval", "Metric", "Current", "Threshold", "Status"]
    parts.append("<table><thead><tr>")
    parts.extend(f"<th>{_esc(h)}</th>" for h in header)
    parts.append("</tr></thead><tbody>")
    for tr in threshold_results:
        current = f"{tr.current_value:.3f}"
        if tr.current_ci:
            lo, hi = tr.current_ci
            current += f" <span class='muted'>[{lo:.3f}, {hi:.3f}]</span>"
        cells = [_esc(tr.eval_name), _esc(tr.metric_name)]
        if has_baselines:
            cells.append(f"{tr.baseline_value:.3f}" if tr.baseline_value is not None else "—")
        cells.append(current)
        cells.append(_esc(_threshold_phrase(tr)))
        cells.append(_status_cell(tr))
        parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    parts.append("</tbody></table>")

    # Regressions
    failed = [tr for tr in threshold_results if not tr.passed]
    if failed:
        parts.append("<h2>Regressions</h2><ul>")
        for tr in failed:
            parts.append(
                f"<li><strong>{_esc(tr.eval_name)} / {_esc(tr.metric_name)}:</strong> "
                f"{_esc(tr.detail)}</li>"
            )
        parts.append("</ul>")

    # Per-example detail
    parts.append("<h2>Examples</h2>")
    for eval_name, trs in by_eval.items():
        result = results_by_name.get(eval_name)
        if result is None or not result.examples:
            continue
        samples_note = (
            f" · {result.samples} rounds" if result.samples > 1 else ""
        )
        parts.append(
            f"<details><summary>{_esc(eval_name)} "
            f"<span class='muted'>({result.num_examples} examples, "
            f"{result.num_errors} errors{samples_note})</span></summary>"
        )
        parts.append(
            "<table><thead><tr><th>Input</th><th>Expected</th><th>Got</th>"
            "<th>Score</th><th>Notes</th></tr></thead><tbody>"
        )
        for i, jr in enumerate(result.per_example):
            ex = result.examples[i] if i < len(result.examples) else None
            res = result.results[i] if i < len(result.results) else None
            if ex is None or res is None:
                continue
            score_cls = "status-pass" if jr.score >= 0.5 else "status-fail"
            notes = res.error or jr.reason or ""
            parts.append(
                "<tr>"
                f"<td>{_esc(_clip(ex.input))}</td>"
                f"<td>{_esc(_clip(ex.expected))}</td>"
                f"<td>{_esc(_clip(res.output))}</td>"
                f"<td class='{score_cls}'>{jr.score:.2f}</td>"
                f"<td class='muted'>{_esc(_clip(notes))}</td>"
                "</tr>"
            )
        parts.append("</tbody></table></details>")

    # Output diffs vs baseline (regressed examples only)
    diff_sections: list[str] = []
    for result in results:
        diffs = compute_output_diffs(result, baselines.get(result.eval_name))
        if not diffs:
            continue
        rows = ["<table><thead><tr><th>Input</th><th>Baseline output</th>"
                "<th>This PR output</th><th>Score</th></tr></thead><tbody>"]
        for d in diffs:
            rows.append(
                "<tr>"
                f"<td>{_esc(_clip(d.input))}</td>"
                f"<td>{_esc(_clip(d.baseline_output))}</td>"
                f"<td>{_esc(_clip(d.current_output))}</td>"
                f"<td class='status-fail'>{d.baseline_score:.2f} &rarr; {d.current_score:.2f}</td>"
                "</tr>"
            )
        rows.append("</tbody></table>")
        diff_sections.append(
            f"<details open><summary>{_esc(result.eval_name)} "
            f"<span class='muted'>({len(diffs)} regressed)</span></summary>"
            + "".join(rows) + "</details>"
        )
    if diff_sections:
        parts.append("<h2>Output Diffs vs Baseline</h2>")
        parts.extend(diff_sections)

    parts.append("</body></html>")
    return "\n".join(parts)


def _clip(text: object, limit: int = 200) -> str:
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


__all__ = [
    "format_report_as", "format_junit", "format_sarif", "format_json",
    "format_html", "VALID_FORMATS",
]
