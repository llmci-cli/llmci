"""Judge calibration and drift detection.

LLM judges drift across model versions and can quietly disagree with humans, which
erodes trust in the gate. ``llmci judge calibrate`` runs a configured judge over a
human-labeled set and reports how well the judge agrees with the labels
(agreement rate, Cohen's kappa, MAE, Pearson correlation).

It also detects *drift*: a calibration snapshot records the judge model and its
per-example scores. When you re-run with a different judge model, the mean absolute
change in scores on the same labeled set is reported so a model swap can't silently
shift the gate.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from llmci.errors import DatasetError
from llmci.judges.base import Judge
from llmci.models import EvalExample, JudgeResult, TargetResult

SNAPSHOT_DIR = Path(".llmci/calibration")


@dataclass
class LabeledExample:
    """A judge calibration example: the output to judge plus human score(s).

    ``criteria`` holds optional per-criterion human scores (e.g. {"faithfulness": 1.0,
    "relevance": 0.0}) for calibrating multi-criterion judges (composite / RAG / safety).
    """

    input: str
    expected: str
    output: str
    human_score: float
    criteria: dict[str, float] = field(default_factory=dict)


def load_labeled_set(path: Path) -> list[LabeledExample]:
    """Load a JSONL labeled set: {input, output, human_score, [expected]} per line.

    ``human_score`` may be a number in [0, 1], a bool, or "pass"/"fail". A line may also
    carry per-criterion labels under ``criteria`` (or ``human_scores``) as a dict; when
    ``human_score`` is omitted but ``criteria`` is present, the overall score is the mean
    of the criterion scores.
    """
    if not path.exists():
        raise DatasetError(f"Labeled set not found: {path}")

    labeled: list[LabeledExample] = []
    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(
                    f"Malformed JSON at {path} line {line_num}: {e}"
                ) from e
            if "input" not in row or "output" not in row:
                raise DatasetError(
                    f"Line {line_num} needs 'input' and 'output' fields."
                )

            raw_criteria = row.get("criteria") or row.get("human_scores")
            criteria: dict[str, float] = {}
            if isinstance(raw_criteria, dict):
                criteria = {k: _normalize_score(v) for k, v in raw_criteria.items()}

            if "human_score" in row:
                human_score = _normalize_score(row["human_score"])
            elif criteria:
                human_score = sum(criteria.values()) / len(criteria)
            else:
                raise DatasetError(
                    f"Line {line_num} needs a 'human_score' or a 'criteria' dict."
                )

            labeled.append(LabeledExample(
                input=row["input"],
                expected=row.get("expected", ""),
                output=row["output"],
                human_score=human_score,
                criteria=criteria,
            ))

    if not labeled:
        raise DatasetError(f"Labeled set is empty: {path}")
    return labeled


def _normalize_score(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("pass", "true", "yes", "1"):
            return 1.0
        if v in ("fail", "false", "no", "0"):
            return 0.0
    raise DatasetError(f"Unrecognized human_score: {value!r}")


@dataclass
class CalibrationResult:
    """Judge↔human agreement metrics plus the raw scores used to compute them."""

    model: str
    n: int
    agreement_rate: float
    cohens_kappa: float
    mae: float
    pearson: float
    judge_scores: list[float] = field(default_factory=list)
    human_scores: list[float] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    # Per-criterion agreement for multi-criterion judges (keyed by criterion name).
    per_criterion: dict[str, "CalibrationResult"] = field(default_factory=dict)


def compute_agreement(
    judge_scores: list[float],
    human_scores: list[float],
) -> tuple[float, float, float, float]:
    """Return (agreement_rate, cohens_kappa, mae, pearson) for paired scores."""
    n = len(judge_scores)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)

    judge_bin = [s >= 0.5 for s in judge_scores]
    human_bin = [s >= 0.5 for s in human_scores]

    agreements = sum(1 for j, h in zip(judge_bin, human_bin) if j == h)
    agreement_rate = agreements / n

    mae = sum(abs(j - h) for j, h in zip(judge_scores, human_scores)) / n
    kappa = _cohens_kappa(judge_bin, human_bin)
    pearson = _pearson(judge_scores, human_scores)
    return (agreement_rate, kappa, mae, pearson)


def _cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_true = sum(a) / n
    pb_true = sum(b) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation; 0.0 when either series has no variance."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    var_x = sum((xi - mx) ** 2 for xi in x)
    var_y = sum((yi - my) ** 2 for yi in y)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


async def run_calibration(
    judge: Judge, model: str, labeled: list[LabeledExample]
) -> CalibrationResult:
    """Run the judge over the labeled set and compute agreement metrics."""
    examples = [EvalExample(input=le.input, expected=le.expected) for le in labeled]
    results = [TargetResult(output=le.output, latency_ms=0.0) for le in labeled]

    judged: list[JudgeResult] = await judge.evaluate_dataset(examples, results)
    judge_scores = [jr.score for jr in judged]
    human_scores = [le.human_score for le in labeled]

    agreement, kappa, mae, pearson = compute_agreement(judge_scores, human_scores)
    return CalibrationResult(
        model=model,
        n=len(labeled),
        agreement_rate=agreement,
        cohens_kappa=kappa,
        mae=mae,
        pearson=pearson,
        judge_scores=judge_scores,
        human_scores=human_scores,
        inputs=[le.input for le in labeled],
        per_criterion=_calibrate_criteria(labeled, judged, model),
    )


def _calibrate_criteria(
    labeled: list[LabeledExample],
    judged: list[JudgeResult],
    model: str,
) -> dict[str, CalibrationResult]:
    """Agreement per criterion, over examples where both human and judge scored it.

    A criterion is calibrated only when humans labeled it (in ``criteria``) and the judge
    surfaced it as a sub-score — so composite/RAG/safety judges get per-criterion trust
    signals without affecting single-score judges (which produce no sub-scores).
    """
    human_criteria: set[str] = set()
    for le in labeled:
        human_criteria.update(le.criteria)

    out: dict[str, CalibrationResult] = {}
    for crit in sorted(human_criteria):
        js: list[float] = []
        hs: list[float] = []
        inputs: list[str] = []
        for le, jr in zip(labeled, judged):
            if crit in le.criteria and jr.sub_scores and crit in jr.sub_scores:
                js.append(jr.sub_scores[crit])
                hs.append(le.criteria[crit])
                inputs.append(le.input)
        if not js:
            continue
        agreement, kappa, mae, pearson = compute_agreement(js, hs)
        out[crit] = CalibrationResult(
            model=model,
            n=len(js),
            agreement_rate=agreement,
            cohens_kappa=kappa,
            mae=mae,
            pearson=pearson,
            judge_scores=js,
            human_scores=hs,
            inputs=inputs,
        )
    return out


@dataclass
class DriftResult:
    """Drift of judge scores vs a stored snapshot from a different model."""

    previous_model: str
    current_model: str
    model_changed: bool
    mean_abs_change: float
    n_compared: int


def snapshot_path(eval_name: str, snapshot_dir: Path | None = None) -> Path:
    return (snapshot_dir or SNAPSHOT_DIR) / f"{eval_name}.json"


def history_path(eval_name: str, snapshot_dir: Path | None = None) -> Path:
    return (snapshot_dir or SNAPSHOT_DIR) / f"{eval_name}-history.jsonl"


def load_snapshot(eval_name: str, snapshot_dir: Path | None = None) -> dict | None:
    path = snapshot_path(eval_name, snapshot_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _history_entry(result: CalibrationResult) -> dict:
    """Serialize a calibration run for the trend history log."""
    entry: dict = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": result.model,
        "n": result.n,
        "agreement_rate": result.agreement_rate,
        "cohens_kappa": result.cohens_kappa,
        "mae": result.mae,
        "pearson": result.pearson,
    }
    if result.per_criterion:
        entry["per_criterion"] = {
            crit: {
                "agreement_rate": cr.agreement_rate,
                "cohens_kappa": cr.cohens_kappa,
                "mae": cr.mae,
            }
            for crit, cr in result.per_criterion.items()
        }
    return entry


def load_history(
    eval_name: str,
    snapshot_dir: Path | None = None,
    *,
    limit: int = 20,
) -> list[dict]:
    """Load prior calibration runs newest-last (chronological order)."""
    path = history_path(eval_name, snapshot_dir)
    if not path.exists():
        return []
    entries: list[dict] = []
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    entries.append(row)
    except (OSError, json.JSONDecodeError):
        return []
    if limit > 0 and len(entries) > limit:
        entries = entries[-limit:]
    return entries


def append_history(
    eval_name: str,
    result: CalibrationResult,
    snapshot_dir: Path | None = None,
) -> Path:
    """Append a calibration run to the trend history log."""
    path = history_path(eval_name, snapshot_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(_history_entry(result)) + "\n")
    return path


def save_snapshot(
    eval_name: str,
    result: CalibrationResult,
    snapshot_dir: Path | None = None,
) -> Path:
    path = snapshot_path(eval_name, snapshot_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": result.model,
        "scores_by_input": dict(zip(result.inputs, result.judge_scores)),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    append_history(eval_name, result, snapshot_dir)
    return path


def compute_drift(result: CalibrationResult, snapshot: dict | None) -> DriftResult | None:
    """Compare current judge scores against a snapshot, matched by input."""
    if not snapshot:
        return None
    prev_scores = snapshot.get("scores_by_input", {})
    prev_model = snapshot.get("model", "unknown")

    paired = [
        (result.judge_scores[i], prev_scores[inp])
        for i, inp in enumerate(result.inputs)
        if inp in prev_scores
    ]
    if not paired:
        return None

    mean_abs_change = sum(abs(cur - prev) for cur, prev in paired) / len(paired)
    return DriftResult(
        previous_model=prev_model,
        current_model=result.model,
        model_changed=prev_model != result.model,
        mean_abs_change=mean_abs_change,
        n_compared=len(paired),
    )


def format_calibration_report(
    result: CalibrationResult,
    drift: DriftResult | None = None,
    history: list[dict] | None = None,
) -> str:
    """Render a human-readable calibration report."""
    lines = [
        "## Judge Calibration",
        "",
        f"Judge model: `{result.model}`  ·  labeled examples: {result.n}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Agreement rate | {result.agreement_rate:.3f} |",
        f"| Cohen's kappa | {result.cohens_kappa:.3f} |",
        f"| Mean abs error | {result.mae:.3f} |",
        f"| Pearson r | {result.pearson:.3f} |",
        "",
        f"_Interpretation: {_kappa_label(result.cohens_kappa)} agreement beyond chance._",
    ]
    if result.per_criterion:
        lines.append("")
        lines.append("### Per-criterion agreement")
        lines.append("")
        lines.append("| Criterion | n | Agreement | Kappa | MAE |")
        lines.append("|-----------|---|-----------|-------|-----|")
        for crit, cr in result.per_criterion.items():
            lines.append(
                f"| {crit} | {cr.n} | {cr.agreement_rate:.3f} | "
                f"{cr.cohens_kappa:.3f} | {cr.mae:.3f} |"
            )
    if drift is not None:
        lines.append("")
        lines.append("### Drift vs snapshot")
        change = "changed" if drift.model_changed else "unchanged"
        lines.append(
            f"Judge model {change} (`{drift.previous_model}` → `{drift.current_model}`); "
            f"mean score change {drift.mean_abs_change:.3f} over {drift.n_compared} examples."
        )
    if history:
        lines.extend(_format_trend_section(history, result))
    return "\n".join(lines)


def _format_trend_section(history: list[dict], current: CalibrationResult) -> list[str]:
    """Render a trend table from prior runs plus the current calibration."""
    rows = list(history)
    current_entry = _history_entry(current)
    if not rows or rows[-1].get("timestamp") != current_entry["timestamp"]:
        rows.append(current_entry)
    if len(rows) < 2:
        return []

    display = rows[-10:]
    lines = [
        "",
        "### Calibration trend",
        "",
        "| Run | Model | Agreement | Kappa | MAE |",
        "|-----|-------|-----------|-------|-----|",
    ]
    for row in display:
        ts = str(row.get("timestamp", "?"))[:16].replace("T", " ")
        model = str(row.get("model", "?"))
        agreement = float(row.get("agreement_rate", 0.0))
        kappa = float(row.get("cohens_kappa", 0.0))
        mae = float(row.get("mae", 0.0))
        lines.append(
            f"| {ts} | `{model}` | {agreement:.3f} | {kappa:.3f} | {mae:.3f} |"
        )
    return lines


def _kappa_label(kappa: float) -> str:
    if kappa < 0.0:
        return "worse-than-chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


__all__ = [
    "LabeledExample", "load_labeled_set", "CalibrationResult", "compute_agreement",
    "run_calibration", "DriftResult", "compute_drift", "save_snapshot",
    "load_snapshot", "load_history", "append_history", "format_calibration_report",
]
