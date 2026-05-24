"""Smoke tests for the CLI skeleton."""

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
    assert "--compare-to" in result.output
    assert "--smoke" in result.output
    assert "--update-baseline" in result.output


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
