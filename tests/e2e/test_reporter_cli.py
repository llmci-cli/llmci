"""End-to-end CLI test for configured report sinks."""

import json
import shlex
import sys
from pathlib import Path

from click.testing import CliRunner

from llmci.cli import cli
from llmci.plugins import reset_registry


def _write_project(root: Path) -> None:
    evals = root / "evals"
    evals.mkdir(parents=True)
    (evals / "d.jsonl").write_text(
        json.dumps({"input": "billing issue", "expected": "billing"}) + "\n"
    )
    (root / "run_target.py").write_text(
        "import argparse, json\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "d = json.loads(open(a.input).read())\n"
        "open(a.output, 'w').write(json.dumps({'output': d['expected']}))\n"
    )
    # Local plugin module: a reporter that writes a sentinel file on each run.
    (root / "sink_plugin.py").write_text(
        "from llmci.plugins import register_reporter\n"
        "def file_sink(ctx):\n"
        "    with open('sink_ran.txt', 'w') as f:\n"
        "        f.write(f'passed={ctx.passed}\\nevals={len(ctx.results)}\\n')\n"
        "        f.write(ctx.report_markdown[:12])\n"
        "register_reporter('file_sink', file_sink)\n"
    )
    python = shlex.quote(sys.executable)
    (root / "llmci.yaml").write_text(
        "version: 1\n"
        "plugins: [sink_plugin]\n"
        "reporters: [file_sink]\n"
        "target:\n"
        f"  command: {python} run_target.py --input {{input_file}} --output {{output_file}}\n"
        "evals:\n"
        "  - name: svc\n"
        "    dataset: ./evals/d.jsonl\n"
        "    judge: exact_match\n"
        "    metrics:\n"
        "      - {name: accuracy, threshold: 1.0, mode: absolute}\n"
    )


def test_configured_reporter_is_invoked():
    runner = CliRunner()
    try:
        with runner.isolated_filesystem():
            _write_project(Path.cwd())
            result = runner.invoke(cli, ["run", "--config", "llmci.yaml"])

            assert result.exit_code == 0, result.output
            sentinel = Path("sink_ran.txt")
            assert sentinel.exists(), "report sink did not run"
            content = sentinel.read_text()
            assert "passed=True" in content
            assert "evals=1" in content
    finally:
        reset_registry()


def test_unregistered_reporter_warns_but_passes():
    runner = CliRunner()
    try:
        with runner.isolated_filesystem():
            root = Path.cwd()
            _write_project(root)
            # Reference a reporter that is never registered.
            cfg = (root / "llmci.yaml").read_text().replace(
                "reporters: [file_sink]", "reporters: [missing_sink]"
            )
            (root / "llmci.yaml").write_text(cfg)

            result = runner.invoke(cli, ["run", "--config", "llmci.yaml"])

            # A missing sink warns but does not fail the gate.
            assert result.exit_code == 0, result.output
            assert "missing_sink" in result.output
            assert not Path("sink_ran.txt").exists()
    finally:
        reset_registry()
