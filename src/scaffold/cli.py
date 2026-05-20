"""Scaffold CLI entry point."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from scaffold import __version__


@click.group()
@click.version_option(version=__version__, prog_name="scaffold")
@click.option("-v", "--verbose", is_flag=True, help="Show progress during eval runs.")
@click.option("--debug", is_flag=True, help="Full debug logging (prompts, responses, timing).")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """Scaffold: CI-native regression testing for LLMs."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug


@cli.command()
@click.option("--compare-to", default=None, help="Branch to compare baselines against.")
@click.option("--smoke", is_flag=True, help="Run on a subset of the dataset.")
@click.option("--output", default=None, type=click.Path(), help="Write report to file.")
@click.option(
    "--update-baseline", is_flag=True, help="Update stored baselines (run on main branch)."
)
@click.option("--seed", default=42, type=int, help="Random seed for smoke sampling.")
@click.pass_context
def run(
    ctx: click.Context,
    compare_to: str | None,
    smoke: bool,
    output: str | None,
    update_baseline: bool,
    seed: int,
) -> None:
    """Run evals and compare against baselines."""
    from scaffold.baseline import load_all_baselines, save_baseline
    from scaffold.config import load_config
    from scaffold.integrations.github import detect_github_context, post_pr_comment
    from scaffold.report import format_report
    from scaffold.runner import run_all_evals

    try:
        config = load_config()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    verbose = ctx.obj.get("verbose", False)
    if verbose:
        click.echo(f"Running {len(config.evals)} eval(s)...")

    try:
        results = asyncio.run(run_all_evals(config, smoke=smoke, seed=seed))
    except Exception as e:
        click.echo(f"Error during eval: {e}", err=True)
        sys.exit(1)

    if update_baseline:
        for result in results:
            path = save_baseline(result)
            if verbose:
                click.echo(f"Baseline saved: {path}")
        click.echo(f"Updated baselines for {len(results)} eval(s).")

    baselines = None
    if compare_to:
        eval_names = [r.eval_name for r in results]
        raw_baselines = load_all_baselines(eval_names, ref=compare_to)
        if raw_baselines:
            baselines = raw_baselines
            if verbose:
                click.echo(f"Loaded {len(baselines)} baseline(s) from {compare_to}")
        elif verbose:
            click.echo(f"No baselines found on {compare_to}")

    report_md, passed = format_report(results, config.evals, baselines=baselines)

    if output:
        Path(output).write_text(report_md)
        if verbose:
            click.echo(f"Report written to {output}")
    else:
        click.echo(report_md)

    gh_ctx = detect_github_context()
    if gh_ctx:
        posted = post_pr_comment(report_md, gh_ctx)
        if verbose:
            if posted:
                click.echo("PR comment posted/updated.")
            else:
                click.echo("Failed to post PR comment.", err=True)

    sys.exit(0 if passed else 1)


@cli.command()
@click.option("--from", "from_model", required=True, help="Source model to migrate from.")
@click.option("--to", "to_model", required=True, help="Target model to migrate to.")
@click.option("--eval", "eval_name", required=True, help="Eval to optimize against.")
@click.option("--optimizer-model", default="gpt-4o", help="Model for prompt optimization.")
@click.option("--patience", default=3, type=int, help="Early stopping patience.")
@click.option("--max-iterations", default=20, type=int, help="Max optimization iterations.")
@click.option(
    "--min-improvement", default=0.005, type=float, help="Min score improvement to reset patience."
)
@click.option(
    "--max-edit-distance",
    default=None,
    type=int,
    help="Reject prompts exceeding this edit distance.",
)
@click.pass_context
def migrate(
    ctx: click.Context,
    from_model: str,
    to_model: str,
    eval_name: str,
    optimizer_model: str,
    patience: int,
    max_iterations: int,
    min_improvement: float,
    max_edit_distance: int | None,
) -> None:
    """Run prompt migration optimization."""
    click.echo("scaffold migrate: not yet implemented (Phase 5)")


@cli.group()
def dataset() -> None:
    """Create and manage eval datasets."""
    pass


@dataset.command("init")
@click.option("--name", required=True, help="Name of the eval dataset.")
@click.option(
    "--type",
    "dataset_type",
    required=True,
    type=click.Choice(["deterministic", "open_ended", "agent"]),
    help="Type of eval dataset.",
)
def dataset_init(name: str, dataset_type: str) -> None:
    """Initialize an empty eval dataset."""
    evals_dir = Path("evals")
    evals_dir.mkdir(exist_ok=True)

    filepath = evals_dir / f"{name}.jsonl"
    if filepath.exists():
        click.echo(f"Dataset already exists: {filepath}")
        return

    filepath.touch()

    guidance = {
        "deterministic": (
            'Add input/expected pairs. Each line: {"input": "...", "expected": "..."}\n'
            "Run `scaffold dataset add --name %s` to add examples interactively."
        ),
        "open_ended": (
            'Add input/expected pairs where expected is a reference response.\n'
            'Each line: {"input": "...", "expected": "..."}\n'
            "Run `scaffold dataset add --name %s` to add examples interactively."
        ),
        "agent": (
            "Add scenario objects with outcomes and constraints.\n"
            "See docs for the agent scenario schema."
        ),
    }

    click.echo(f"Created {filepath}")
    click.echo(f"\nType: {dataset_type}")
    click.echo(guidance.get(dataset_type, "") % name)


@dataset.command("add")
@click.option("--name", required=True, help="Name of the eval dataset.")
def dataset_add(name: str) -> None:
    """Add examples interactively."""
    import json

    filepath = Path("evals") / f"{name}.jsonl"
    if not filepath.exists():
        click.echo(
            f"Dataset not found: {filepath}\n"
            f"Run `scaffold dataset init --name {name} --type deterministic` first."
        )
        return

    count = sum(1 for line in filepath.read_text().splitlines() if line.strip())
    click.echo(f"Adding examples to {filepath} ({count} existing)\n")

    while True:
        input_text = click.prompt("Input", type=str)
        expected_text = click.prompt("Expected", type=str)

        row = json.dumps({"input": input_text, "expected": expected_text})
        with filepath.open("a") as f:
            f.write(row + "\n")

        count += 1
        click.echo(f"  Added. ({count} examples total)")

        if not click.confirm("\nAdd another?", default=True):
            break

    click.echo(f"\nDone. {count} examples in {filepath}.")


@dataset.command("check")
@click.option("--name", required=True, help="Name of the eval dataset.")
def dataset_check(name: str) -> None:
    """Analyze dataset coverage and quality."""
    click.echo("scaffold dataset check: not yet implemented (Phase 4)")


@dataset.command("import")
@click.option("--name", required=True, help="Name of the eval dataset.")
@click.option("--from", "source", required=True, help="Path to CSV or JSON file.")
def dataset_import(name: str, source: str) -> None:
    """Import examples from CSV or JSON."""
    click.echo("scaffold dataset import: not yet implemented (Phase 4)")


@cli.command()
def init() -> None:
    """Initialize a new scaffold.yaml interactively."""
    click.echo("scaffold init: not yet implemented (Phase 7)")


@cli.command("import-promptfoo")
@click.argument("source", type=click.Path(exists=True))
@click.option("--output", default="scaffold.yaml", help="Output path for converted config.")
def import_promptfoo(source: str, output: str) -> None:
    """Convert a Promptfoo config to scaffold.yaml."""
    click.echo("scaffold import-promptfoo: not yet implemented (Phase 7)")
