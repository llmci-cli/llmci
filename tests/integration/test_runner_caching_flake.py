"""Integration tests for response caching and flake resistance through the runner.

Both features previously had only unit coverage. These drive a direct (litellm) target
through ``run_all_evals`` with a mocked provider, exercising the cache wiring and the
multi-sample aggregation / confidence-interval path end to end.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from llmci.cache import ResponseCache
from llmci.config import load_config
from llmci.runner import run_all_evals


def _make_response(content: str):
    """Build a minimal litellm-shaped response object."""
    usage = type("U", (), {"prompt_tokens": 5, "completion_tokens": 7})()
    message = type("M", (), {"content": content})()
    choice = type("C", (), {"message": message})()
    return type(
        "R",
        (),
        {"choices": [choice], "usage": usage, "_hidden_params": {"response_cost": 0.001}},
    )()


def _write_direct_config(tmp_path: Path, *, rows: list[tuple[str, str]], extra: str = "") -> None:
    (tmp_path / "evals").mkdir(exist_ok=True)
    lines = "".join(
        f'{{"input": {i!r}, "expected": {e!r}}}\n'.replace("'", '"')
        for i, e in rows
    )
    (tmp_path / "evals" / "d.jsonl").write_text(lines)
    (tmp_path / "llmci.yaml").write_text(
        "version: 1\n"
        "target:\n  provider: openai\n  model: gpt-4o-mini\n"
        "evals:\n"
        "  - name: direct\n"
        "    dataset: ./evals/d.jsonl\n"
        "    judge: {type: exact_match}\n"
        "    metrics:\n"
        "      - {name: mean_score, threshold: 0.0, mode: absolute}\n"
        f"{extra}"
    )


@pytest.mark.asyncio
async def test_response_cache_serves_second_run(tmp_path, monkeypatch):
    """Two runs sharing a cache: the second is fully served from disk (no LLM calls)."""
    _write_direct_config(tmp_path, rows=[("a", "a"), ("b", "b"), ("c", "c")])
    monkeypatch.chdir(tmp_path)

    calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        calls["n"] += 1
        # Echo the prompt (== input) so exact_match passes; content is cacheable.
        return _make_response(kwargs["messages"][0]["content"])

    cache = ResponseCache(cache_dir=tmp_path / ".llmci" / "cache", enabled=True)
    config = load_config()

    with patch("llmci.targets.direct.litellm.acompletion", side_effect=fake_acompletion):
        first = await run_all_evals(config, cache=cache)
        assert calls["n"] == 3  # 3 examples => 3 provider calls
        assert cache.misses == 3 and cache.hits == 0

        second = await run_all_evals(config, cache=cache)
        assert calls["n"] == 3  # no new provider calls on the second run
        assert cache.hits == 3

    assert first[0].metrics["mean_score"] == 1.0
    assert second[0].metrics["mean_score"] == 1.0


@pytest.mark.asyncio
async def test_judge_cache_serves_second_run(tmp_path, monkeypatch):
    """An LLM judge's calls are cached across runs: the second run makes no judge calls."""
    import shlex
    import sys

    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "d.jsonl").write_text(
        '{"input": "is this ok?"}\n{"input": "and this?"}\n'
    )
    # Deterministic command target (not cached itself) so only judge calls hit litellm.
    py = shlex.quote(sys.executable)
    (tmp_path / "run.py").write_text(
        "import argparse, json\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "d = json.loads(open(a.input).read())\n"
        # Distinct answer per input so each example is a distinct judge prompt.
        "open(a.output, 'w').write(json.dumps({'output': 'a calm reply to ' + d['input']}))\n"
    )
    (tmp_path / "llmci.yaml").write_text(
        "version: 1\n"
        f"target:\n  command: {py} run.py --input {{input_file}} --output {{output_file}}\n"
        "evals:\n"
        "  - name: safe\n"
        "    dataset: ./evals/d.jsonl\n"
        "    judge: {type: safety, criteria: [{name: toxicity, type: toxicity}]}\n"
        "    metrics:\n"
        "      - {name: toxicity, threshold: 0.5, mode: absolute}\n"
    )
    monkeypatch.chdir(tmp_path)

    calls = {"n": 0}

    async def fake_judge(**kwargs):
        calls["n"] += 1
        return _make_response('{"score": 1.0, "reasoning": "safe"}')

    cache = ResponseCache(cache_dir=tmp_path / ".llmci" / "cache", enabled=True)
    config = load_config()

    with patch("llmci.judges.llm_cache.litellm.acompletion", side_effect=fake_judge):
        first = await run_all_evals(config, cache=cache)
        assert calls["n"] == 2  # 2 examples => 2 judge calls

        second = await run_all_evals(config, cache=cache)
        assert calls["n"] == 2  # second run fully served from the judge cache

    assert first[0].metrics["toxicity"] == 1.0
    assert second[0].metrics["toxicity"] == 1.0


@pytest.mark.asyncio
async def test_flake_resistance_aggregates_with_ci(tmp_path, monkeypatch):
    """samples_per_example > 1 averages rounds and reports a confidence interval."""
    # parallelism: 1 makes the mocked call order deterministic across rounds.
    _write_direct_config(
        tmp_path,
        rows=[("q1", "yes"), ("q2", "yes")],
        extra="settings:\n  samples_per_example: 3\n  parallelism: 1\n  significance: 0.95\n",
    )
    monkeypatch.chdir(tmp_path)

    # 2 examples x 3 rounds = 6 calls. Scores per round: 1.0, 0.5, 0.0 -> mean 0.5.
    sequence = ["yes", "yes", "yes", "no", "no", "no"]
    calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        content = sequence[calls["n"]]
        calls["n"] += 1
        return _make_response(content)

    config = load_config()
    with patch("llmci.targets.direct.litellm.acompletion", side_effect=fake_acompletion):
        results = await run_all_evals(config)

    result = results[0]
    assert calls["n"] == 6  # cache bypassed when sampling
    assert result.samples == 3
    assert result.metrics["mean_score"] == pytest.approx(0.5)
    # A confidence interval is reported and has non-zero width given the spread.
    assert "mean_score" in result.metric_ci
    low, high = result.metric_ci["mean_score"]
    assert low < result.metrics["mean_score"] < high
