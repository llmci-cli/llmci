"""Integration tests for agent examples."""

import asyncio
from pathlib import Path

from scaffold.config import load_config
from scaffold.runner import run_all_evals

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


class TestAgentSingleTurn:
    def test_single_turn_example(self, monkeypatch):
        monkeypatch.chdir(EXAMPLES_DIR / "05-agent-single-turn")
        config = load_config()
        results = asyncio.run(run_all_evals(config))

        assert len(results) == 1
        r = results[0]
        assert r.eval_name == "agent-tool-use"
        assert r.num_examples == 5
        assert r.num_errors == 0
        assert 0.0 <= r.metrics["mean_score"] <= 1.0
        assert 0.0 <= r.metrics["pass_rate"] <= 1.0


class TestAgentMultiTurn:
    def test_multi_turn_example(self, monkeypatch):
        monkeypatch.chdir(EXAMPLES_DIR / "06-agent-multi-turn")
        config = load_config()
        results = asyncio.run(run_all_evals(config))

        assert len(results) == 1
        r = results[0]
        assert r.eval_name == "customer-support-conversation"
        assert r.num_examples == 2
        assert r.num_errors == 0
        assert 0.0 <= r.metrics["mean_score"] <= 1.0
