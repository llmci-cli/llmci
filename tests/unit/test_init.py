"""Tests for llmci init."""

import json

from click.testing import CliRunner

from llmci.cli import cli


class TestLlmciInit:
    def test_creates_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["init"], input="command\nclassification\nmy-eval\n\n")
        assert result.exit_code == 0
        assert (tmp_path / "llmci.yaml").exists()
        assert (tmp_path / "evals" / "my-eval.jsonl").exists()

    def test_creates_classification_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["init"], input="command\nclassification\ntest-eval\n\n")
        assert result.exit_code == 0

        import yaml
        config = yaml.safe_load((tmp_path / "llmci.yaml").read_text())
        assert config["version"] == 1
        assert config["evals"][0]["name"] == "test-eval"
        assert config["evals"][0]["judge"] == "exact_match"

    def test_creates_open_ended_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["init"],
            input="command\nopen_ended\ngen-eval\ngpt-4o-mini\n\n",
        )
        assert result.exit_code == 0

        import yaml
        config = yaml.safe_load((tmp_path / "llmci.yaml").read_text())
        assert config["evals"][0]["judge"]["type"] == "llm"

    def test_creates_agent_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["init"], input="command\nagent\nagent-eval\n\n")
        assert result.exit_code == 0

        import yaml
        config = yaml.safe_load((tmp_path / "llmci.yaml").read_text())
        assert config["evals"][0]["level"] == "agent"
        assert config["evals"][0]["judge"]["type"] == "composite"

    def test_creates_starter_dataset(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init"], input="command\nclassification\nmy-eval\n\n")

        lines = (tmp_path / "evals" / "my-eval.jsonl").read_text().strip().splitlines()
        assert len(lines) >= 3
        for line in lines:
            row = json.loads(line)
            assert "input" in row
            assert "expected" in row

    def test_aborts_on_existing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "llmci.yaml").write_text("existing")
        runner = CliRunner()
        result = runner.invoke(cli, ["init"], input="n\n")
        assert "Aborted" in result.output

    def test_direct_mode(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["init"],
            input="direct\nclassification\nmy-eval\nopenai\ngpt-4o-mini\n",
        )
        assert result.exit_code == 0

        import yaml
        config = yaml.safe_load((tmp_path / "llmci.yaml").read_text())
        assert "direct" in config["target"]
