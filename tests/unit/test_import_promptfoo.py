"""Tests for Promptfoo config import."""

import json

import pytest
import yaml

from llmci.errors import ConfigError
from llmci.import_promptfoo import import_promptfoo_config


class TestImportPromptfoo:
    def test_basic_conversion(self, tmp_path):
        source = tmp_path / "promptfooconfig.yaml"
        source.write_text(yaml.dump({
            "providers": ["openai:gpt-4o"],
            "tests": [
                {
                    "vars": {"input": "hello"},
                    "assert": [{"type": "equals", "value": "greeting"}],
                },
            ],
        }))

        output = tmp_path / "llmci.yaml"
        import_promptfoo_config(source, output)

        config = yaml.safe_load(output.read_text())
        assert config["version"] == 1
        assert config["target"]["direct"]["provider"] == "openai"
        assert config["target"]["direct"]["model"] == "gpt-4o"

    def test_creates_dataset(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        source = tmp_path / "promptfooconfig.yaml"
        source.write_text(yaml.dump({
            "providers": ["openai:gpt-4o"],
            "tests": [
                {
                    "vars": {"input": "hello"},
                    "assert": [{"type": "equals", "value": "hi"}],
                },
                {
                    "vars": {"input": "bye"},
                    "assert": [{"type": "equals", "value": "goodbye"}],
                },
            ],
        }))

        output = tmp_path / "llmci.yaml"
        import_promptfoo_config(source, output)

        dataset_path = tmp_path / "evals" / "imported.jsonl"
        assert dataset_path.exists()
        lines = dataset_path.read_text().strip().splitlines()
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert row["input"] == "hello"
        assert row["expected"] == "hi"

    def test_llm_rubric_detection(self, tmp_path):
        source = tmp_path / "promptfooconfig.yaml"
        source.write_text(yaml.dump({
            "providers": ["openai:gpt-4o"],
            "tests": [
                {
                    "vars": {"input": "explain X"},
                    "assert": [{"type": "llm-rubric", "value": "clear explanation"}],
                },
            ],
        }))

        output = tmp_path / "llmci.yaml"
        import_promptfoo_config(source, output)

        config = yaml.safe_load(output.read_text())
        assert config["evals"][0]["judge"]["type"] == "llm"

    def test_missing_source(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            import_promptfoo_config(tmp_path / "missing.yaml")

    def test_invalid_yaml(self, tmp_path):
        source = tmp_path / "bad.yaml"
        source.write_text(": invalid: yaml: [")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            import_promptfoo_config(source, tmp_path / "out.yaml")

    def test_no_providers(self, tmp_path):
        source = tmp_path / "promptfooconfig.yaml"
        source.write_text(yaml.dump({"tests": []}))

        output = tmp_path / "llmci.yaml"
        import_promptfoo_config(source, output)

        config = yaml.safe_load(output.read_text())
        assert "command" in config["target"]

    def test_prompt_extraction(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        source = tmp_path / "promptfooconfig.yaml"
        source.write_text(yaml.dump({
            "providers": ["openai:gpt-4o"],
            "prompts": ["You are a helpful assistant. Answer: {{input}}"],
            "tests": [],
        }))

        output = tmp_path / "llmci.yaml"
        import_promptfoo_config(source, output)

        config = yaml.safe_load(output.read_text())
        assert config["target"].get("prompt_file") == "prompt.txt"
        assert (tmp_path / "prompt.txt").exists()
