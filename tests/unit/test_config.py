"""Tests for config parsing and validation."""

from pathlib import Path

import pytest
import yaml

from llmci.config import _normalize_judge, load_config
from llmci.errors import ConfigError
from llmci.models import DatasetSource


@pytest.fixture
def tmp_config(tmp_path):
    """Helper to write a YAML config and return its path."""
    def _write(data: dict) -> Path:
        p = tmp_path / "llmci.yaml"
        p.write_text(yaml.dump(data))
        return p
    return _write


class TestLoadConfig:
    def test_minimal_command_config(self, tmp_config):
        path = tmp_config({
            "version": 1,
            "target": {"command": "echo test --input {input_file} --output {output_file}"},
            "evals": [{
                "name": "test-eval",
                "dataset": "./data.jsonl",
                "judge": "exact_match",
                "metrics": [{"name": "accuracy", "threshold": 0.9, "mode": "absolute"}],
            }],
        })
        config = load_config(path)
        assert config.version == 1
        assert config.target.is_command_mode
        assert len(config.evals) == 1
        assert config.evals[0].judge.type == "exact_match"

    def test_minimal_direct_config(self, tmp_config):
        path = tmp_config({
            "version": 1,
            "target": {"provider": "openai", "model": "gpt-4o"},
            "evals": [{
                "name": "test-eval",
                "dataset": "./data.jsonl",
                "metrics": [{"name": "accuracy", "threshold": 0.9, "mode": "absolute"}],
            }],
        })
        config = load_config(path)
        assert not config.target.is_command_mode
        assert config.target.provider == "openai"
        assert config.target.model == "gpt-4o"

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "llmci.yaml"
        p.write_text(":\ninvalid: [yaml")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(p)

    def test_no_target(self, tmp_config):
        path = tmp_config({
            "version": 1,
            "evals": [{"name": "test", "dataset": "d.jsonl", "metrics": []}],
        })
        with pytest.raises(ConfigError):
            load_config(path)

    def test_both_command_and_direct(self, tmp_config):
        path = tmp_config({
            "version": 1,
            "target": {"command": "echo", "provider": "openai", "model": "gpt-4o"},
            "evals": [{"name": "test", "dataset": "d.jsonl", "metrics": []}],
        })
        with pytest.raises(ConfigError):
            load_config(path)

    def test_agent_level_rejected(self, tmp_config):
        path = tmp_config({
            "version": 1,
            "target": {"command": "echo"},
            "evals": [{
                "name": "agent-eval",
                "level": "agent",
                "dataset": "d.jsonl",
                "metrics": [],
            }],
        })
        with pytest.raises(ConfigError, match="agent"):
            load_config(path)

    def test_settings_defaults(self, tmp_config):
        path = tmp_config({
            "version": 1,
            "target": {"command": "echo"},
            "evals": [{"name": "test", "dataset": "d.jsonl", "metrics": []}],
        })
        config = load_config(path)
        assert config.settings.parallelism == 10
        assert config.settings.timeout_per_call == 30
        assert config.settings.retries == 2

    def test_custom_settings(self, tmp_config):
        path = tmp_config({
            "version": 1,
            "target": {"command": "echo"},
            "evals": [{"name": "test", "dataset": "d.jsonl", "metrics": []}],
            "settings": {"parallelism": 5, "timeout_per_call": 60},
        })
        config = load_config(path)
        assert config.settings.parallelism == 5
        assert config.settings.timeout_per_call == 60

    def test_remote_dataset_source(self, tmp_config):
        path = tmp_config({
            "version": 1,
            "target": {"command": "echo"},
            "evals": [{
                "name": "remote-eval",
                "dataset": {
                    "source": "s3://company-evals/tickets.jsonl",
                    "cache": True,
                },
                "metrics": [],
            }],
        })
        config = load_config(path)
        dataset = config.evals[0].dataset
        assert isinstance(dataset, DatasetSource)
        assert dataset.source == "s3://company-evals/tickets.jsonl"
        assert dataset.cache is True

    def test_remote_dataset_s3_string(self, tmp_config):
        path = tmp_config({
            "version": 1,
            "target": {"command": "echo"},
            "evals": [{
                "name": "remote-eval",
                "dataset": "s3://company-evals/tickets.jsonl",
                "metrics": [],
            }],
        })
        config = load_config(path)
        assert config.evals[0].dataset == "s3://company-evals/tickets.jsonl"


class TestNormalizeJudge:
    def test_string_exact_match(self):
        result = _normalize_judge("exact_match")
        assert result == {"type": "exact_match"}

    def test_string_llm(self):
        result = _normalize_judge("llm")
        assert result == {"type": "llm"}

    def test_invalid_string(self):
        with pytest.raises(ConfigError, match="Unknown judge shorthand"):
            _normalize_judge("invalid_judge")

    def test_dict_passthrough(self):
        raw = {"type": "llm", "model": "gpt-4o"}
        result = _normalize_judge(raw)
        assert result == raw
