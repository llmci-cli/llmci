"""Tests for agent scenario loading."""

import json
from pathlib import Path

import pytest

from scaffold.dataset.loader import load_agent_scenarios
from scaffold.errors import DatasetError


def _write_scenarios(path: Path, scenarios: list[dict]):
    with path.open("w") as f:
        for s in scenarios:
            f.write(json.dumps(s) + "\n")


class TestLoadAgentScenarios:
    def test_single_turn(self, tmp_path):
        p = tmp_path / "scenarios.jsonl"
        _write_scenarios(p, [
            {
                "input": {"query": "hello"},
                "expected": {"outcome": "greeting"},
            },
        ])
        scenarios = load_agent_scenarios(p)
        assert len(scenarios) == 1
        assert not scenarios[0].is_multi_turn

    def test_multi_turn(self, tmp_path):
        p = tmp_path / "scenarios.jsonl"
        _write_scenarios(p, [
            {
                "turns": [
                    {"user_message": "hi", "expected": {"outcome": "greeting"}},
                    {"user_message": "bye", "expected": {"outcome": "farewell"}},
                ],
            },
        ])
        scenarios = load_agent_scenarios(p)
        assert len(scenarios) == 1
        assert scenarios[0].is_multi_turn

    def test_missing_file(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_agent_scenarios(tmp_path / "missing.jsonl")

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        with pytest.raises(DatasetError, match="empty"):
            load_agent_scenarios(p)

    def test_malformed_json(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n")
        with pytest.raises(DatasetError, match="Malformed"):
            load_agent_scenarios(p)

    def test_invalid_scenario(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        _write_scenarios(p, [{"foo": "bar"}])
        with pytest.raises(DatasetError, match="Invalid agent scenario"):
            load_agent_scenarios(p)

    def test_smoke_sampling(self, tmp_path):
        p = tmp_path / "scenarios.jsonl"
        _write_scenarios(p, [
            {"input": f"q{i}", "expected": {"outcome": f"a{i}"}}
            for i in range(20)
        ])
        scenarios = load_agent_scenarios(p, smoke_size=5)
        assert len(scenarios) == 5

    def test_with_constraints(self, tmp_path):
        p = tmp_path / "scenarios.jsonl"
        _write_scenarios(p, [
            {
                "input": "test",
                "expected": {
                    "outcome": "result",
                    "constraints": {
                        "max_tool_calls": 3,
                        "required_tools": ["search"],
                    },
                },
            },
        ])
        scenarios = load_agent_scenarios(p)
        assert scenarios[0].expected.constraints.max_tool_calls == 3
