"""llmci CLI entry point."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from fnmatch import fnmatch
from pathlib import Path

import click

from llmci import __version__

IGNORED_CONFIG_DIRS = {
    ".git",
    ".hg",
    ".llmci",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "venv",
}


def _matches_any(path: Path, patterns: tuple[str, ...]) -> bool:
    """Return whether a path matches any shell-style pattern."""
    path_text = path.as_posix()
    return any(fnmatch(path_text, pattern) for pattern in patterns)


def discover_config_files(
    root: Path = Path("."),
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> list[Path]:
    """Find llmci config files under root."""
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    cwd = Path.cwd().resolve()
    configs: list[Path] = []
    for path in root.rglob("llmci*.yaml"):
        rel_to_root = path.relative_to(root)
        if any(part in IGNORED_CONFIG_DIRS for part in rel_to_root.parts[:-1]):
            continue
        try:
            config_path = path.relative_to(cwd)
        except ValueError:
            config_path = path

        if include and not _matches_any(config_path, include):
            continue
        if exclude and _matches_any(config_path, exclude):
            continue

        configs.append(config_path)

    return sorted(configs, key=lambda p: str(p))


@click.group()
@click.version_option(version=__version__, prog_name="llmci")
@click.option("-v", "--verbose", is_flag=True, help="Show progress during eval runs.")
@click.option("--debug", is_flag=True, help="Full debug logging (prompts, responses, timing).")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """llmci: CI-native regression testing for LLMs."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug


def _run_config(
    ctx: click.Context,
    config_path: Path,
    compare_to: str | None,
    smoke: bool,
    output: str | None,
    update_baseline: bool,
    seed: int,
    post_github: bool = True,
    output_format: str = "markdown",
    cache_enabled: bool = True,
    refresh_cache: bool = False,
    samples: int | None = None,
    significance: float | None = None,
) -> int:
    """Run a single config and return the process exit code."""
    from llmci.baseline import load_all_baselines, save_baseline
    from llmci.cache import ResponseCache
    from llmci.config import load_config
    from llmci.integrations.github import detect_github_context, post_pr_comment
    from llmci.report import format_report
    from llmci.report_formats import format_report_as
    from llmci.runner import run_all_evals

    original_cwd = Path.cwd()
    resolved_config_path = (
        config_path if config_path.is_absolute() else original_cwd / config_path
    ).resolve()
    config_dir = resolved_config_path.parent
    config_name = Path(resolved_config_path.name)
    output_path = (
        (Path(output) if Path(output).is_absolute() else original_cwd / output).resolve()
        if output
        else None
    )
    exit_code = 1

    if not resolved_config_path.exists():
        click.echo(
            f"Error: Config file not found: {config_path}\n\n"
            "Fix: Run 'llmci init' to create a llmci.yaml, "
            "or create one manually.",
            err=True,
        )
        return 1

    try:
        os.chdir(config_dir)

        try:
            config = load_config(config_name)
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            return 1

        if samples is not None:
            config.settings.samples_per_example = samples
        if significance is not None:
            config.settings.significance = significance

        verbose = ctx.obj.get("verbose", False)
        if verbose:
            click.echo(f"Running {len(config.evals)} eval(s)...")

        cache = ResponseCache(enabled=cache_enabled, refresh=refresh_cache)

        try:
            results = asyncio.run(
                run_all_evals(config, smoke=smoke, seed=seed, cache=cache)
            )
        except Exception as e:
            click.echo(f"Error during eval: {e}", err=True)
            return 1

        if verbose and cache_enabled and (cache.hits or cache.misses):
            click.echo(
                f"Response cache: {cache.hits} hit(s), {cache.misses} miss(es)"
            )

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

        if output_format == "markdown":
            rendered = report_md
        else:
            rendered, _ = format_report_as(
                output_format, results, config.evals, baselines=baselines
            )

        if output_path:
            output_path.write_text(rendered)
            if verbose:
                click.echo(f"Report written to {output_path}")
        else:
            click.echo(rendered)

        gh_ctx = detect_github_context() if post_github else None
        if gh_ctx:
            from llmci.integrations.github import resolve_report_slice_key

            slice_key = resolve_report_slice_key()
            posted = post_pr_comment(report_md, gh_ctx, slice_key=slice_key)
            if verbose:
                if posted:
                    click.echo("PR comment posted/updated.")
                else:
                    click.echo("Failed to post PR comment.", err=True)

        exit_code = 0 if passed else 1
    finally:
        os.chdir(original_cwd)

    return exit_code


@cli.command()
@click.option(
    "--config",
    "config_path",
    default="llmci.yaml",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Path to llmci config file.",
)
@click.option("--all", "run_all_configs", is_flag=True, help="Run all discovered configs.")
@click.option(
    "--root",
    default=".",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    help="Directory to search when using --all.",
)
@click.option(
    "--include",
    multiple=True,
    help="Only run discovered config paths matching this glob. May be used multiple times.",
)
@click.option(
    "--exclude",
    multiple=True,
    help="Skip discovered config paths matching this glob. May be used multiple times.",
)
@click.option("--compare-to", default=None, help="Branch to compare baselines against.")
@click.option("--smoke", is_flag=True, help="Run on a subset of the dataset.")
@click.option("--output", default=None, type=click.Path(), help="Write report to file.")
@click.option(
    "--output-format",
    type=click.Choice(["markdown", "junit", "sarif", "json"]),
    default="markdown",
    help="Report format for --output/stdout. PR comments stay markdown.",
)
@click.option(
    "--update-baseline", is_flag=True, help="Update stored baselines (run on main branch)."
)
@click.option("--seed", default=42, type=int, help="Random seed for smoke sampling.")
@click.option(
    "--no-cache", is_flag=True, help="Disable the direct-target response cache."
)
@click.option(
    "--refresh-cache",
    is_flag=True,
    help="Ignore cached responses on read but write fresh results back.",
)
@click.option(
    "--samples",
    default=None,
    type=int,
    help="Run each eval N rounds for flake resistance (overrides settings).",
)
@click.option(
    "--significance",
    default=None,
    type=float,
    help="Confidence level (e.g. 0.95) for significance-gated regressions.",
)
@click.pass_context
def run(
    ctx: click.Context,
    config_path: Path,
    run_all_configs: bool,
    root: Path,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    compare_to: str | None,
    smoke: bool,
    output: str | None,
    output_format: str,
    update_baseline: bool,
    seed: int,
    no_cache: bool,
    refresh_cache: bool,
    samples: int | None,
    significance: float | None,
) -> None:
    """Run evals and compare against baselines."""
    if run_all_configs and output:
        click.echo("Error: --output cannot be used with --all.", err=True)
        sys.exit(1)

    if run_all_configs:
        try:
            config_paths = discover_config_files(root, include=include, exclude=exclude)
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        if not config_paths:
            click.echo("No llmci config files found.", err=True)
            sys.exit(1)

        exit_code = 0
        for index, path in enumerate(config_paths):
            if index:
                click.echo()
            click.echo(f"### {path}")
            result_code = _run_config(
                ctx,
                path,
                compare_to=compare_to,
                smoke=smoke,
                output=None,
                update_baseline=update_baseline,
                seed=seed,
                post_github=False,
                output_format=output_format,
                cache_enabled=not no_cache,
                refresh_cache=refresh_cache,
                samples=samples,
                significance=significance,
            )
            if result_code != 0:
                exit_code = result_code

        sys.exit(exit_code)

    sys.exit(_run_config(
        ctx,
        config_path,
        compare_to=compare_to,
        smoke=smoke,
        output=output,
        update_baseline=update_baseline,
        seed=seed,
        output_format=output_format,
        cache_enabled=not no_cache,
        refresh_cache=refresh_cache,
        samples=samples,
        significance=significance,
    ))


@cli.command()
@click.option(
    "--root",
    default=".",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    help="Directory to search.",
)
@click.option("--json", "json_output", is_flag=True, help="Output config paths as JSON.")
@click.option(
    "--include",
    multiple=True,
    help="Only list config paths matching this glob. May be used multiple times.",
)
@click.option(
    "--exclude",
    multiple=True,
    help="Skip config paths matching this glob. May be used multiple times.",
)
def discover(
    root: Path,
    json_output: bool,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> None:
    """Discover llmci config files in a repository."""
    try:
        config_paths = discover_config_files(root, include=include, exclude=exclude)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    path_strings = [str(path) for path in config_paths]
    if json_output:
        click.echo(json.dumps(path_strings, indent=2))
        return

    if not path_strings:
        click.echo("No llmci config files found.")
        return

    for path in path_strings:
        click.echo(path)


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
    from llmci.config import find_eval, load_config
    from llmci.dataset.loader import load_dataset
    from llmci.migrate.optimizer import MigrationProgressEvent, optimize_prompt
    from llmci.migrate.report import format_migration_report
    from llmci.migrate.splitter import split_dataset

    try:
        config = load_config()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    eval_cfg = find_eval(config, eval_name)

    if not config.target.prompt_file:
        click.echo(
            "Error: Migration requires a prompt file. "
            "Set target.prompt_file in llmci.yaml.",
            err=True,
        )
        sys.exit(1)

    original_prompt = Path(config.target.prompt_file).read_text()
    examples = load_dataset(eval_cfg.dataset)
    split = split_dataset(examples)
    primary_metric = eval_cfg.metrics[0].name if eval_cfg.metrics else "accuracy"

    click.echo(
        f"Migrating {from_model} → {to_model}\n"
        f"Eval: {eval_name} (metric: {primary_metric})\n"
        f"Split: {len(split.train)} train / {len(split.validation)} val / "
        f"{len(split.holdout)} holdout\n"
        f"Optimizer: {optimizer_model}\n"
    )

    def report_progress(event: MigrationProgressEvent) -> None:
        if event.phase == "baseline_complete" and event.original_score is not None:
            click.echo(f"Baseline holdout on old model: {event.original_score:.3f}")
        elif event.phase == "initial_train_complete" and event.train_score is not None:
            click.echo(f"Initial train score on target model: {event.train_score:.3f}")
        elif event.phase == "iteration_start":
            click.echo(
                f"\nIteration {event.iteration}/{event.max_iterations}\n"
                f"  failures: {event.failure_count}\n"
                f"  current train: {_format_optional_score(event.train_score)}"
            )
        elif event.phase == "iteration_complete":
            accepted = "yes" if event.accepted else "no"
            click.echo(
                f"  train: {_format_optional_score(event.train_score)}\n"
                f"  val: {_format_optional_score(event.val_score)}\n"
                f"  accepted: {accepted}"
            )
        elif event.phase == "iteration_skipped":
            click.echo(f"  skipped: {event.reason or 'unknown'}")
        elif event.phase == "complete" and event.holdout_score is not None:
            click.echo(f"\nFinal holdout on target model: {event.holdout_score:.3f}\n")

    try:
        result = asyncio.run(optimize_prompt(
            original_prompt=original_prompt,
            from_model=from_model,
            to_model=to_model,
            optimizer_model=optimizer_model,
            eval_config=eval_cfg,
            split=split,
            primary_metric=primary_metric,
            patience=patience,
            min_improvement=min_improvement,
            max_iterations=max_iterations,
            max_edit_distance=max_edit_distance,
            base_url=config.target.base_url,
            progress_callback=report_progress,
        ))
    except Exception as e:
        click.echo(f"Error during migration: {e}", err=True)
        sys.exit(1)

    click.echo(format_migration_report(result))

    prompt_path = Path(config.target.prompt_file)
    if result.best_prompt != original_prompt:
        if click.confirm("\nWrite optimized prompt to disk?"):
            prompt_path.write_text(result.best_prompt)
            click.echo(f"Prompt written to {prompt_path}")


def _format_optional_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.3f}"


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
            "Run `llmci dataset add --name %s` to add examples interactively."
        ),
        "open_ended": (
            'Add input/expected pairs where expected is a reference response.\n'
            'Each line: {"input": "...", "expected": "..."}\n'
            "Run `llmci dataset add --name %s` to add examples interactively."
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
            f"Run `llmci dataset init --name {name} --type deterministic` first."
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
@click.option(
    "--min-per-category",
    default=5,
    type=int,
    help="Minimum examples per category before warning.",
)
def dataset_check(name: str, min_per_category: int) -> None:
    """Analyze dataset coverage and quality."""
    from llmci.dataset.check import check_dataset, format_coverage_report

    filepath = Path("evals") / f"{name}.jsonl"
    if not filepath.exists():
        click.echo(f"Dataset not found: {filepath}")
        sys.exit(1)

    report = check_dataset(filepath, min_per_category=min_per_category)
    click.echo(format_coverage_report(report))


@dataset.command("import")
@click.option("--name", required=True, help="Name of the eval dataset.")
@click.option("--from", "source", required=True, help="Path to CSV or JSON file.")
@click.option("--input-column", default="input", help="Column name for input.")
@click.option("--expected-column", default="expected", help="Column name for expected.")
def dataset_import(
    name: str, source: str, input_column: str, expected_column: str
) -> None:
    """Import examples from CSV or JSON."""
    from llmci.dataset.import_data import import_dataset

    try:
        imported, skipped = import_dataset(
            name, source, input_column=input_column, expected_column=expected_column
        )
        click.echo(f"Imported {imported} examples to evals/{name}.jsonl")
        if skipped:
            click.echo(f"Skipped {skipped} rows (missing input or expected)")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def init() -> None:
    """Initialize a new llmci.yaml interactively."""
    from llmci.init import run_init

    run_init()


@cli.command("import-promptfoo")
@click.argument("source", type=click.Path(exists=True))
@click.option("--output", default="llmci.yaml", help="Output path for converted config.")
def import_promptfoo(source: str, output: str) -> None:
    """Convert a Promptfoo config to llmci.yaml."""
    from llmci.import_promptfoo import import_promptfoo_config

    try:
        import_promptfoo_config(Path(source), Path(output))
        click.echo(f"Converted {source} → {output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
