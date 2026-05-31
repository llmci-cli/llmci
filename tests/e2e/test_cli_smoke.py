"""Smoke tests for the CLI skeleton."""

import json
import shlex
import sys
from pathlib import Path

from click.testing import CliRunner

from llmci import __version__
from llmci.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "CI-native regression testing" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_run_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--all" in result.output
    assert "--config" in result.output
    assert "--compare-to" in result.output
    assert "--root" in result.output
    assert "--smoke" in result.output
    assert "--update-baseline" in result.output


def write_service_config(service_dir: Path, config_name: str, eval_name: str) -> None:
    evals_dir = service_dir / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir / "examples.jsonl").write_text(
        json.dumps({"input": "billing issue", "expected": "billing"}) + "\n"
    )
    (service_dir / "run_target.py").write_text(
        "import argparse, json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--input')\n"
        "parser.add_argument('--output')\n"
        "args = parser.parse_args()\n"
        "data = json.loads(open(args.input).read())\n"
        "open(args.output, 'w').write(json.dumps({'output': data['expected']}))\n"
    )
    python = shlex.quote(sys.executable)
    (service_dir / config_name).write_text(
        "version: 1\n"
        "target:\n"
        f"  command: {python} run_target.py --input {{input_file}} --output {{output_file}}\n"
        "evals:\n"
        f"  - name: {eval_name}\n"
        "    dataset: ./evals/examples.jsonl\n"
        "    judge: exact_match\n"
        "    metrics:\n"
        "      - name: accuracy\n"
        "        threshold: 1.0\n"
        "        mode: absolute\n"
    )


def test_run_config_from_nested_service_directory():
    runner = CliRunner()
    with runner.isolated_filesystem():
        original_cwd = Path.cwd()
        write_service_config(Path("services/api"), "llmci-alt.yaml", "nested-service")
        result = runner.invoke(cli, ["run", "--config", "services/api/llmci-alt.yaml"])

        assert result.exit_code == 0
        assert "nested-service" in result.output
        assert Path.cwd() == original_cwd


def test_discover_lists_configs_and_skips_ignored_dirs():
    runner = CliRunner()
    with runner.isolated_filesystem():
        write_service_config(Path("services/api"), "llmci.yaml", "api-service")
        write_service_config(Path("services/agent"), "llmci-agent.yaml", "agent-service")
        write_service_config(Path(".llmci/cache"), "llmci.yaml", "cached-service")
        write_service_config(Path("node_modules/pkg"), "llmci.yaml", "vendor-service")

        result = runner.invoke(cli, ["discover"])

        assert result.exit_code == 0
        assert result.output.splitlines() == [
            "services/agent/llmci-agent.yaml",
            "services/api/llmci.yaml",
        ]


def test_run_all_discovered_configs():
    runner = CliRunner()
    with runner.isolated_filesystem():
        write_service_config(Path("services/api"), "llmci.yaml", "api-service")
        write_service_config(Path("services/agent"), "llmci-agent.yaml", "agent-service")

        result = runner.invoke(cli, ["run", "--all"])

        assert result.exit_code == 0
        assert "### services/agent/llmci-agent.yaml" in result.output
        assert "### services/api/llmci.yaml" in result.output
        assert "agent-service" in result.output
        assert "api-service" in result.output


def test_run_all_uses_root_option():
    runner = CliRunner()
    with runner.isolated_filesystem():
        write_service_config(Path("services/api"), "llmci.yaml", "api-service")
        write_service_config(Path("services/agent"), "llmci-agent.yaml", "agent-service")

        result = runner.invoke(cli, ["run", "--all", "--root", "services/api"])

        assert result.exit_code == 0
        assert "### services/api/llmci.yaml" in result.output
        assert "api-service" in result.output
        assert "agent-service" not in result.output


def test_migrate_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["migrate", "--help"])
    assert result.exit_code == 0
    assert "--from" in result.output
    assert "--to" in result.output
    assert "--eval" in result.output


def test_dataset_subcommands():
    runner = CliRunner()
    result = runner.invoke(cli, ["dataset", "--help"])
    assert result.exit_code == 0
    for cmd in ["init", "add", "check", "import"]:
        assert cmd in result.output


def test_discover_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["discover", "--help"])
    assert result.exit_code == 0
    assert "--root" in result.output


def test_init_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["init", "--help"])
    assert result.exit_code == 0


def test_import_promptfoo_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["import-promptfoo", "--help"])
    assert result.exit_code == 0
