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
    assert "--config" in result.output
    assert "--compare-to" in result.output
    assert "--smoke" in result.output
    assert "--update-baseline" in result.output


def test_run_config_from_nested_service_directory():
    runner = CliRunner()
    with runner.isolated_filesystem():
        original_cwd = Path.cwd()
        service_dir = Path("services/api")
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
        (service_dir / "llmci-alt.yaml").write_text(
            "version: 1\n"
            "target:\n"
            f"  command: {python} run_target.py --input {{input_file}} --output {{output_file}}\n"
            "evals:\n"
            "  - name: nested-service\n"
            "    dataset: ./evals/examples.jsonl\n"
            "    judge: exact_match\n"
            "    metrics:\n"
            "      - name: accuracy\n"
            "        threshold: 1.0\n"
            "        mode: absolute\n"
        )

        result = runner.invoke(cli, ["run", "--config", "services/api/llmci-alt.yaml"])

        assert result.exit_code == 0
        assert "nested-service" in result.output
        assert Path.cwd() == original_cwd


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


def test_init_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["init", "--help"])
    assert result.exit_code == 0


def test_import_promptfoo_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["import-promptfoo", "--help"])
    assert result.exit_code == 0
