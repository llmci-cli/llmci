"""Integration tests for the trickier judge features.

Covers judge calibration (deterministic, via the 14-judge-calibration example), the
pairwise judge driven through the real runner with a stored baseline and a mocked LLM,
and the machine-readable report formats rendered from a real eval result.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import pytest

from llmci.config import load_config
from llmci.runner import run_all_evals

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


class _MockMessage:
    def __init__(self, content):
        self.content = content


class _MockResponse:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _MockMessage(content)})()]


def _mock_llm(content):
    async def _m(*args, **kwargs):
        return _MockResponse(content)

    return _m


def _mock_prefer(winning_text):
    """Position-independent pairwise mock: picks whichever side has winning_text."""

    async def _m(*args, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        a_section = prompt[prompt.index("## Answer A") : prompt.index("## Answer B")]
        winner = "A" if winning_text in a_section else "B"
        return _MockResponse(f'{{"winner": "{winner}", "reasoning": "content"}}')

    return _m


@pytest.mark.asyncio
async def test_calibration_example(monkeypatch):
    """14-judge-calibration: exact_match judge vs human labels, deterministic."""
    monkeypatch.chdir(EXAMPLES_DIR / "14-judge-calibration")

    from llmci.calibrate import load_labeled_set, run_calibration
    from llmci.config import find_eval
    from llmci.judges.factory import create_judge

    config = load_config()
    eval_cfg = find_eval(config, "ticket-classification")
    judge = create_judge(eval_cfg.judge)
    labeled = load_labeled_set(Path("labels.jsonl"))

    result = await run_calibration(judge, eval_cfg.judge.model or "default", labeled)

    assert result.n == 6
    # 5 of 6 labels agree with the exact_match judge (one human-accepted miss).
    assert result.agreement_rate == pytest.approx(5 / 6)
    assert result.cohens_kappa == pytest.approx(0.5714, abs=0.01)
    assert result.mae == pytest.approx(1 / 6)


@pytest.mark.asyncio
async def test_pairwise_through_runner(tmp_path, monkeypatch):
    """Pairwise judging end-to-end: baseline load -> set_baseline -> compare -> win_rate."""
    (tmp_path / "evals").mkdir()
    # q1/q2 have baseline outputs; q3 is new (no baseline -> neutral 0.5).
    (tmp_path / "evals" / "q.jsonl").write_text(
        '{"input": "q1"}\n{"input": "q2"}\n{"input": "q3"}\n'
    )
    (tmp_path / "llmci.yaml").write_text(
        "version: 1\n"
        "target:\n  command: \"printf 'new answer' > {output_file}\"\n"
        "evals:\n"
        "  - name: pw\n"
        "    dataset: ./evals/q.jsonl\n"
        "    judge: {type: pairwise, model: gpt-4o-mini}\n"
        "    metrics:\n"
        "      - {name: win_rate, threshold: 0.5, mode: absolute}\n"
    )
    baseline_dir = tmp_path / ".llmci" / "baselines"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "pw.json").write_text(json.dumps({
        "eval_name": "pw",
        "metrics": {"win_rate": 0.5},
        "timestamp": "2026-01-01T00:00:00+00:00",
        "commit_sha": "abc123",
        "examples": [
            {"input": "q1", "output": "old answer 1", "score": 0.5},
            {"input": "q2", "output": "old answer 2", "score": 0.5},
        ],
    }))

    monkeypatch.chdir(tmp_path)

    from llmci.baseline import load_all_baselines

    config = load_config()
    baselines = load_all_baselines(["pw"])

    # The current output ("new answer") wins in either position -> a real win that
    # survives the default position-swap averaging.
    with patch("llmci.judges.pairwise.litellm.acompletion",
               side_effect=_mock_prefer("new answer")):
        results = await run_all_evals(config, baselines=baselines)

    result = results[0]
    assert result.num_examples == 3
    scores = [jr.score for jr in result.per_example]
    assert scores[0] == 1.0  # q1 current wins
    assert scores[1] == 1.0  # q2 current wins
    assert scores[2] == 0.5  # q3 has no baseline -> neutral
    assert result.metrics["win_rate"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)


@pytest.mark.asyncio
async def test_report_formats_on_real_example(monkeypatch):
    """JUnit/SARIF/JSON/HTML render from a real eval result and stay well-formed."""
    monkeypatch.chdir(EXAMPLES_DIR / "04-custom-judge")
    config = load_config()
    results = await run_all_evals(config)

    from llmci.report_formats import format_report_as

    junit, passed = format_report_as("junit", results, config.evals)
    assert passed
    root = ET.fromstring(junit)
    assert root.tag == "testsuites"

    sarif, _ = format_report_as("sarif", results, config.evals)
    sarif_obj = json.loads(sarif)
    assert "runs" in sarif_obj
    assert sarif_obj["version"] == "2.1.0"

    json_report, _ = format_report_as("json", results, config.evals)
    parsed = json.loads(json_report)
    assert parsed  # non-empty, valid JSON

    html, _ = format_report_as("html", results, config.evals)
    assert "<html" in html
    assert "api-json-validation" in html
