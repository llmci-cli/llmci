# Scaffold: Implementation Plan

This document is the build guide. It specifies what to build, in what order, and what "done" looks like for each phase. Interfaces and data structures are defined concretely so that each module can be built and tested independently.

**Companion doc:** `PLAN.md` covers the *what* and *why*. This doc covers the *how*.

---

## Guiding Principles

1. **Smallest useful thing first.** Each phase should produce something a real user could run. No phase depends on "we'll finish it later."
2. **Interfaces before internals.** Define the contracts between modules first. Implementations can be swapped.
3. **Test the tool with the tool.** Once Phase 1 works, we use Scaffold to eval Scaffold's own judges (meta-testing).
4. **Config is the product.** The YAML schema and CLI flags are the primary UX. Every design decision should be evaluated against "is this easy to write in the config?"
5. **Command mode is the default path.** Direct API mode is a convenience. The command (black-box) path must always work and be first-class.

---

## Phase 0: Project Scaffolding

**Goal:** A repo with the right structure, dependencies, packaging, and a CLI skeleton that does nothing but respond to `--help`.

### 0.1 Repo Initialization

```
scaffold-ai/
├── pyproject.toml
├── README.md
├── LICENSE                         # Apache 2.0
├── .github/
│   └── workflows/
│       └── ci.yml                  # lint + test on every PR
├── src/
│   └── scaffold/
│       ├── __init__.py
│       ├── cli.py
│       └── ...                     # empty modules per package structure in PLAN.md
├── tests/
│   ├── conftest.py
│   └── ...
└── examples/                       # empty for now, populated in Phase 5
```

### 0.2 Dependencies

```toml
# pyproject.toml
[project]
name = "scaffold-ai"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "click>=8.1",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "litellm>=1.40",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "mypy>=1.10",
]

[project.scripts]
scaffold = "scaffold.cli:cli"
```

**Why these choices:**
- **click** over typer: more mature, fewer surprises with subcommands, wider adoption.
- **pydantic** for config validation: strict typing, good error messages, serialization for baselines.
- **litellm** for LLM calls: 100+ provider support, env var routing, zero auth management.
- **rich** for terminal output: tables, progress bars, colored diffs in reports.

### 0.3 CLI Skeleton

```python
# src/scaffold/cli.py
import click

@click.group()
@click.version_option()
@click.option("-v", "--verbose", is_flag=True, help="Show progress during eval runs.")
@click.option("--debug", is_flag=True, help="Full debug logging (prompts, responses, timing).")
@click.pass_context
def cli(ctx, verbose, debug):
    """Scaffold: CI-native regression testing for LLMs."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug

@cli.command()
@click.option("--compare-to", default=None, help="Branch to compare baselines against.")
@click.option("--smoke", is_flag=True, help="Run on a subset of the dataset.")
@click.option("--output", default=None, help="Write report to file.")
@click.option("--update-baseline", is_flag=True, help="Update stored baselines (run on main).")
@click.option("--seed", default=42, help="Random seed for smoke sampling.")
def run(compare_to, smoke, output, update_baseline, seed):
    """Run evals and compare against baselines."""
    click.echo("scaffold run: not yet implemented")

@cli.command()
@click.option("--from", "from_model", required=True, help="Source model to migrate from.")
@click.option("--to", "to_model", required=True, help="Target model to migrate to.")
@click.option("--eval", "eval_name", required=True, help="Eval to optimize against.")
@click.option("--optimizer-model", default="gpt-4o", help="Model for prompt optimization.")
@click.option("--patience", default=3, help="Early stopping patience.")
@click.option("--max-iterations", default=20, help="Max optimization iterations.")
@click.option("--min-improvement", default=0.005, help="Min score improvement to reset patience.")
@click.option("--max-edit-distance", default=None, type=int, help="Reject prompts exceeding this edit distance.")
def migrate(from_model, to_model, eval_name, optimizer_model, patience, max_iterations, min_improvement, max_edit_distance):
    """Run prompt migration optimization."""
    click.echo("scaffold migrate: not yet implemented")

@cli.group()
def dataset():
    """Create and manage eval datasets."""
    pass

@dataset.command("init")
@click.option("--name", required=True)
@click.option("--type", "dataset_type", type=click.Choice(["deterministic", "open_ended", "agent"]), required=True)
def dataset_init(name, dataset_type):
    """Initialize an empty eval dataset."""
    click.echo("scaffold dataset init: not yet implemented")

@dataset.command("add")
@click.option("--name", required=True)
def dataset_add(name):
    """Add examples interactively."""
    click.echo("scaffold dataset add: not yet implemented")

@dataset.command("check")
@click.option("--name", required=True)
def dataset_check(name):
    """Analyze dataset coverage and quality."""
    click.echo("scaffold dataset check: not yet implemented")

@dataset.command("import")
@click.option("--name", required=True)
@click.option("--from", "source", required=True, help="Path to CSV or JSON file.")
def dataset_import(name, source):
    """Import examples from CSV or JSON."""
    click.echo("scaffold dataset import: not yet implemented")

@cli.command()
def init():
    """Initialize a new scaffold.yaml interactively."""
    click.echo("scaffold init: not yet implemented")

@cli.command("import-promptfoo")
@click.argument("source", type=click.Path(exists=True))
@click.option("--output", default="scaffold.yaml", help="Output path for converted config.")
def import_promptfoo(source, output):
    """Convert a Promptfoo config to scaffold.yaml."""
    click.echo("scaffold import-promptfoo: not yet implemented")
```

### 0.4 Acceptance Criteria

- [ ] `pip install -e .` works
- [ ] `scaffold --help` shows all commands and subcommands
- [ ] `scaffold run --help` shows all flags
- [ ] `scaffold --version` prints version
- [ ] `ruff check` and `mypy` pass
- [ ] CI runs lint + tests on PR

---

## Phase 1: Core Eval Loop

**Goal:** Run a deterministic eval locally. `scaffold run` reads a config, loads a dataset, calls the target, evaluates with a judge, and prints a pass/fail report.

This is the "hello world" — the minimum that demonstrates the tool works.

### 1.1 Data Models

All data models use Pydantic for validation, serialization, and clear error messages.

```python
# src/scaffold/models.py
from pydantic import BaseModel, Field, model_validator
from typing import Literal
from pathlib import Path

class TargetConfig(BaseModel):
    command: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_file: Path | None = None

    @model_validator(mode="after")
    def validate_mode(self):
        has_command = self.command is not None
        has_direct = self.provider is not None or self.model is not None
        if has_command and has_direct:
            raise ValueError("Specify either 'command' or 'provider'+'model', not both")
        if not has_command and not has_direct:
            raise ValueError("Specify either 'command' or 'provider'+'model'")
        return self

class MetricThreshold(BaseModel):
    name: str
    threshold: float
    mode: Literal["absolute", "max_regression"]

class RubricCriterion(BaseModel):
    id: str
    prompt: str

class JudgeConfig(BaseModel):
    type: Literal["exact_match", "llm", "custom", "composite"] = "exact_match"
    model: str | None = None                    # for type=llm
    rubric: list[RubricCriterion] | str | None = None  # list for v1 llm, string for v2 composite criteria
    module: str | None = None                   # for type=custom
    function: str | None = None                 # for type=custom
    criteria: list[dict] | None = None          # for type=composite (Phase 6)

class DatasetSource(BaseModel):
    """Remote dataset source (v2). V1 uses plain string paths only."""
    source: str
    cache: bool = True

class EvalConfig(BaseModel):
    name: str
    level: Literal["prompt", "pipeline", "agent"] = "pipeline"
    dataset: str | DatasetSource
    target: TargetConfig | None = None          # per-eval override
    judge: JudgeConfig                          # always normalized by load_config
    metrics: list[MetricThreshold]
    mode: Literal["full_replay", "history_injection"] | None = None  # agent multi-turn only

class Settings(BaseModel):
    parallelism: int = 10
    timeout_per_call: int = 30
    retries: int = 2
    smoke_test_size: int | None = None

class ScaffoldConfig(BaseModel):
    version: int = 1
    target: TargetConfig
    evals: list[EvalConfig]
    settings: Settings = Settings()
```

**Design notes:**
- **`judge` is always `JudgeConfig` after parsing.** The config parser (`load_config`) normalizes string shorthands like `judge: exact_match` into `JudgeConfig(type="exact_match")` before constructing the model. Raw YAML accepts strings; the Pydantic model only accepts the normalized form.
- **No separate `metrics` judge type.** The `judge` determines how to score *each example* (exact match, LLM rubric, custom function). Aggregate metrics (F1, accuracy, precision, recall) are always computed from the per-example scores, regardless of judge type. This means `judge: exact_match` + `metrics: [{name: f1_macro, ...}]` is valid — exact match produces per-example 0/1 labels, and the metrics layer computes F1 from those labels.
- **`TargetConfig` validates mutual exclusivity** — command mode xor direct mode.
- **`level: "agent"` and `mode`** are reserved for Phase 6. V1 config parser rejects `level: agent` with a clear "agent eval requires v2" error.
- **`DatasetSource`** is defined now so the schema accommodates remote sources without a breaking change, but v1 `load_config` rejects non-string datasets.

### 1.2 Config Parser

```python
# src/scaffold/config.py

def load_config(path: Path = Path("scaffold.yaml")) -> ScaffoldConfig:
    """Load and validate scaffold.yaml. Raises ConfigError with human-readable messages."""
    ...

def normalize_judge(raw: str | dict) -> JudgeConfig:
    """Convert shorthand judge strings to full JudgeConfig objects."""
    ...
```

**Acceptance criteria:**
- [ ] Parses the full example config from PLAN.md without error
- [ ] Rejects invalid configs with clear error messages (missing required fields, unknown judge types, etc.)
- [ ] Normalizes `judge: exact_match` → `JudgeConfig(type="exact_match")`
- [ ] Validates that command mode has `command` set and direct mode has `provider` + `model`

### 1.3 Dataset Loader

```python
# src/scaffold/dataset.py
from pydantic import BaseModel, Field

class EvalExample(BaseModel):
    input: str
    expected: str
    extra: dict = Field(default_factory=dict)   # pass-through fields

def load_dataset(path: Path, smoke_size: int | None = None, seed: int = 42) -> list[EvalExample]:
    """Load JSONL dataset. Optionally subsample for smoke tests (deterministic seed)."""
    ...
```

**Acceptance criteria:**
- [ ] Loads JSONL files, one JSON object per line
- [ ] Validates that every line has `input` and `expected`
- [ ] Extra fields preserved in `extra` dict (any field beyond `input`/`expected`)
- [ ] `smoke_size` selects a deterministic random subset (fixed seed)
- [ ] Clear error on malformed lines (line number + content)

**Input file contract for command mode:** when the target runner writes the temp input file, it writes the **full JSONL row as JSON** — the entire `{"input": "...", "expected": "...", ...}` object. The customer's command reads this JSON and extracts what it needs. This passes through all `extra` fields naturally. The output file should contain a JSON object with at minimum an `"output"` field: `{"output": "hardware"}`. Additional fields (latency, metadata) are preserved but optional.

### 1.4 Target Runner

The target runner executes the customer's LLM pipeline for each example and collects outputs.

```python
# src/scaffold/targets/command.py
from dataclasses import dataclass

@dataclass
class TargetResult:
    output: str
    latency_ms: float
    error: str | None = None

async def run_command_target(
    command_template: str,
    examples: list[EvalExample],
    parallelism: int = 10,
    timeout: int = 30,
    retries: int = 2,
) -> list[TargetResult]:
    """
    For each example:
    1. Write input to a temp file
    2. Execute the command as a subprocess (with {input_file} and {output_file} substituted)
    3. Read the output file
    4. Return TargetResult

    Runs up to `parallelism` examples concurrently using asyncio.
    Retries on timeout or non-zero exit code.
    """
    ...
```

```python
# src/scaffold/targets/direct.py

async def run_direct_target(
    provider: str,
    model: str,
    prompt_template: str,
    examples: list[EvalExample],
    parallelism: int = 10,
    timeout: int = 30,
    retries: int = 2,
) -> list[TargetResult]:
    """
    For each example:
    1. Substitute {input} into prompt_template
    2. Call litellm.acompletion(model=f"{provider}/{model}", ...)
    3. Return TargetResult with the response text

    Uses asyncio semaphore for parallelism.
    """
    ...
```

**Acceptance criteria:**
- [ ] Command mode: writes temp input file, runs subprocess, reads output file, cleans up
- [ ] Command mode: handles non-zero exit codes, timeouts, missing output files
- [ ] Direct mode: calls litellm with correct provider/model routing
- [ ] Both modes: respects parallelism limit, retries on failure
- [ ] Latency tracked per call

### 1.5 Judges (Deterministic Only)

Phase 1 ships two judges. LLM-as-judge and custom judges come in Phase 3.

```python
# src/scaffold/judges/base.py
from dataclasses import dataclass

@dataclass
class JudgeResult:
    score: float              # 0.0 to 1.0
    reason: str | None = None

class Judge:
    """Base interface for all judges. All methods are async for uniformity
    (sync judges just don't await anything internally)."""

    async def evaluate_single(self, input: str, expected: str, actual: str) -> JudgeResult:
        """Score a single example. Returns 0.0–1.0."""
        raise NotImplementedError

    async def evaluate_dataset(
        self, examples: list[EvalExample], results: list[TargetResult]
    ) -> tuple[list[JudgeResult], dict[str, float]]:
        """
        Score all examples and compute aggregate metrics.

        Returns:
            per_example: one JudgeResult per row (for degraded-examples report)
            metrics: metric_name → score (for threshold checking)

        Default implementation calls evaluate_single on each pair, then computes
        aggregate metrics from the per-example scores. Subclasses can override
        for dataset-level metrics (e.g., F1 requires the full label distribution).
        """
        per_example = [
            await self.evaluate_single(ex.input, ex.expected, r.output)
            for ex, r in zip(examples, results)
            if r.error is None
        ]
        return per_example, self._aggregate(per_example, examples, results)

    def _aggregate(self, per_example, examples, results) -> dict[str, float]:
        """Compute standard aggregate metrics from per-example scores."""
        ...
```

**The metrics layer** is built into every judge, not a separate judge type:

```python
# src/scaffold/metrics.py

def compute_metrics(
    examples: list[EvalExample],
    results: list[TargetResult],
    per_example: list[JudgeResult],
    requested: list[str],
) -> dict[str, float]:
    """
    Compute requested aggregate metrics from per-example results.

    Always available (from per-example scores):
      - pass_rate: fraction with score >= 0.5
      - mean_score: average score

    Available when expected/actual are categorical labels:
      - accuracy: exact match fraction
      - f1_macro, f1_micro, f1_weighted
      - precision_macro, recall_macro

    The runner calls this after the judge to fill EvalResult.metrics
    with only the metrics the user asked for in their thresholds.
    """
    ...
```

This resolves the "exact_match judge + f1_macro metric" problem: the judge produces per-example 0/1 scores, and `compute_metrics` derives F1 from the expected/actual label pairs.

```python
# src/scaffold/judges/exact_match.py

class ExactMatchJudge(Judge):
    """Score is 1.0 if output == expected (stripped, optionally case-insensitive), else 0.0."""

    def __init__(self, case_sensitive: bool = True):
        self.case_sensitive = case_sensitive

    async def evaluate_single(self, input, expected, actual) -> JudgeResult:
        ...
```

**Acceptance criteria:**
- [ ] `ExactMatchJudge` correctly handles string stripping and case sensitivity
- [ ] `compute_metrics` computes F1 macro/micro/weighted correctly on multi-class data
- [ ] `compute_metrics` handles edge cases: single-class data, empty predictions, all-wrong predictions
- [ ] Target errors (non-None `TargetResult.error`) are excluded from judge scoring and counted separately
- [ ] Per-example JudgeResults are stored for the degraded-examples report

### 1.6 Runner (Orchestrator)

The runner ties everything together for a single eval.

```python
# src/scaffold/runner.py

@dataclass
class EvalResult:
    eval_name: str
    metrics: dict[str, float]           # metric_name → score
    per_example: list[JudgeResult]      # one per non-error dataset row
    examples: list[EvalExample]         # corresponding examples (for degraded-examples report)
    results: list[TargetResult]         # corresponding target outputs
    latency_stats: dict[str, float]     # p50, p90, p99, mean
    num_examples: int
    num_errors: int

def resolve_target(eval_config: EvalConfig, global_target: TargetConfig) -> TargetConfig:
    """Per-eval target overrides global target. Eval target wins if present."""
    return eval_config.target or global_target

async def run_eval(
    eval_config: EvalConfig,
    target_config: TargetConfig,
    settings: Settings,
) -> EvalResult:
    """
    Execute one eval end to end:
    1. Load dataset (subsample if --smoke)
    2. Resolve target (per-eval override or global)
    3. Run target on all examples → list[TargetResult]
    4. Run judge on all examples → list[JudgeResult]
    5. Compute requested metrics from per-example results
    6. Return EvalResult with everything needed for reporting
    """
    target = resolve_target(eval_config, target_config)
    examples = load_dataset(eval_config.dataset, settings.smoke_test_size)
    results = await run_target(target, examples, settings)
    judge = create_judge(eval_config.judge)
    per_example, _ = await judge.evaluate_dataset(examples, results)
    requested_metrics = [m.name for m in eval_config.metrics]
    metrics = compute_metrics(examples, results, per_example, requested_metrics)
    ...

async def run_all_evals(config: ScaffoldConfig) -> list[EvalResult]:
    """Run all evals in the config sequentially (evals are independent)."""
    ...
```

**Acceptance criteria:**
- [ ] Runs a single eval with exact_match judge and produces correct metrics
- [ ] Runs multiple evals from one config
- [ ] Per-eval target override works (eval target beats global target)
- [ ] Reports errors (target failures) separately from judge failures
- [ ] Target failures are excluded from metrics computation (not scored as 0)
- [ ] Latency stats computed from TargetResult timing

### 1.7 Report Generator (Phase 1: Local Only)

Phase 1 report handles the no-baseline case: absolute thresholds only, printed to stdout.

```python
# src/scaffold/report.py

def format_report(
    results: list[EvalResult],
    configs: list[EvalConfig],
    baselines: dict[str, dict[str, float]] | None = None,
) -> tuple[str, bool]:
    """
    Returns (markdown_string, all_passed).

    Phase 1 (no baselines):
    - Summary table: eval | metric | score | threshold | status
    - Only absolute thresholds are checked
    - max_regression thresholds are skipped with a warning ("no baseline found, run --update-baseline first")
    - Failed examples listed (examples where judge score < 0.5)

    Phase 2 (with baselines):
    - Summary table adds baseline column: eval | metric | baseline | this run | threshold | status
    - max_regression thresholds are checked: (baseline - current) / baseline <= threshold
    - Degraded examples: inputs that failed on this PR (not "changed from pass to fail" — we don't store per-example baselines)
    """
    ...
```

**Why no per-example baselines:** storing per-example pass/fail state would require running the full eval on the baseline branch and comparing row-by-row. This is expensive and fragile (dataset changes between branches). Instead, "degraded examples" means "examples that fail on this PR" — always useful for debugging, regardless of whether they also failed on main.

**Acceptance criteria:**
- [ ] Produces a readable markdown table
- [ ] Correctly evaluates absolute thresholds (score >= threshold)
- [ ] Without baselines: max_regression thresholds are skipped with a clear warning
- [ ] Returns `all_passed=False` if any absolute threshold is violated
- [ ] Lists failed examples (score < 0.5) with input, expected, and actual output
- [ ] Exit code 0 on pass, 1 on fail wired up in CLI

### 1.8 Wiring It All Up

Update `cli.py` to connect the real implementations:

```python
@cli.command()
def run(...):
    config = load_config()
    results = asyncio.run(run_all_evals(config))
    report_md, passed = format_report(results, ...)
    click.echo(report_md)
    sys.exit(0 if passed else 1)
```

### 1.9 Dataset Commands (Basic)

PLAN includes basic dataset commands in Phase 1 to lower the adoption barrier from day one. These are lightweight CLI tools — not the coverage analysis or import features from Phase 4.

```python
# src/scaffold/dataset/init_cmd.py
def init_dataset(name: str, dataset_type: str, base_dir: Path = Path("evals")):
    """Create an empty JSONL file and print guidance on next steps."""
    ...

# src/scaffold/dataset/add_cmd.py
def add_example(name: str, base_dir: Path = Path("evals")):
    """Interactive loop: prompt for input/expected, append to JSONL, show count."""
    ...
```

These two commands are simple and self-contained. `scaffold dataset check` and `scaffold dataset import` come in Phase 4 since they require more logic (coverage analysis, format parsing).

### 1.10 Phase 1 Acceptance (End-to-End)

- [ ] Create `examples/01-ci-regression/` with a real scaffold.yaml, dataset, and run_prompt.py
- [ ] `scaffold run` in that directory (command mode) produces a correct report
- [ ] `scaffold run` in direct API mode also works (with a real or mocked LLM)
- [ ] Changing the prompt or model in the example causes different scores
- [ ] `--smoke` runs a deterministic subset
- [ ] Exit code is 1 when absolute thresholds are violated, 0 when they pass
- [ ] max_regression thresholds are skipped with a warning when no baseline exists
- [ ] `--output report.md` writes to file
- [ ] Error handling: missing config file, missing dataset, command timeout, malformed JSONL all produce clear error messages
- [ ] `scaffold dataset init --name test --type deterministic` creates an empty JSONL file
- [ ] `scaffold dataset add --name test` interactively adds examples

---

## Phase 2: Baselines + CI

**Goal:** Compare PR results against a stored main-branch baseline and post the result as a CI check. This is where Scaffold becomes a real team tool.

### 2.1 Baseline Storage

```python
# src/scaffold/baseline.py

BASELINE_DIR = Path(".scaffold/baselines")

@dataclass
class Baseline:
    eval_name: str
    metrics: dict[str, float]
    timestamp: str
    commit_sha: str

def save_baseline(result: EvalResult, commit_sha: str) -> Path:
    """
    Write baseline to .scaffold/baselines/{eval_name}.json
    Called by `scaffold run --update-baseline`.
    """
    ...

def load_baseline(eval_name: str, ref: str | None = None) -> Baseline | None:
    """
    Load baseline for an eval.

    If ref is provided (e.g., "main", "origin/main"), load from that git ref
    using `git show {ref}:.scaffold/baselines/{eval_name}.json`.

    If ref is None, load from the local filesystem.

    Returns None if no baseline exists (first run).
    """
    ...

def load_all_baselines(eval_names: list[str], ref: str | None = None) -> dict[str, Baseline]:
    """Load baselines for all evals. Missing baselines are omitted."""
    ...
```

**Design notes:**
- Baselines are JSON files committed to the repo under `.scaffold/baselines/`.
- On the main branch, CI runs `scaffold run --update-baseline` which saves new baselines.
- On PR branches, `scaffold run --compare-to=origin/main` loads baselines from the base branch via `git show`.
- This is fully stateless — no external service, no database.

**Acceptance criteria:**
- [ ] `--update-baseline` writes JSON files to `.scaffold/baselines/`
- [ ] `--compare-to=main` loads baselines from the git ref
- [ ] Missing baselines are handled gracefully (first run, new eval added)
- [ ] Stale baselines (eval renamed/removed) don't cause crashes
- [ ] `git show` failure (ref doesn't exist) produces a clear error

### 2.2 Regression Detection

```python
# src/scaffold/comparison.py

@dataclass
class ThresholdResult:
    eval_name: str
    metric_name: str
    baseline_value: float | None
    current_value: float
    threshold: float
    mode: str                       # "absolute" or "max_regression"
    passed: bool
    detail: str                     # human-readable explanation

def check_thresholds(
    results: list[EvalResult],
    baselines: dict[str, Baseline],
    configs: list[EvalConfig],
) -> list[ThresholdResult]:
    """
    For each eval × metric:
    - absolute: current >= threshold
    - max_regression: (baseline - current) / baseline <= threshold

    Returns one ThresholdResult per eval × metric pair.
    """
    ...
```

**Acceptance criteria:**
- [ ] Absolute thresholds checked correctly
- [ ] Relative thresholds checked correctly (percentage drop, not absolute difference)
- [ ] Handles edge cases: baseline is 0 (avoid division by zero), baseline is None (skip relative check)
- [ ] Detail string is clear: "dropped 2.9% (0.972 → 0.943, threshold: 2%)"

### 2.3 PR Report (Full)

Extend the Phase 1 report to include baseline comparison, degraded examples, and a clear pass/fail verdict.

```python
# src/scaffold/report.py

def format_pr_report(
    results: list[EvalResult],
    threshold_results: list[ThresholdResult],
    degraded_examples: dict[str, list[dict]],
) -> tuple[str, bool]:
    """
    Full PR report as markdown:

    ## Scaffold Eval Report

    | Eval | Metric | Baseline | This PR | Threshold | Status |
    |------|--------|----------|---------|-----------|--------|
    | ...  | ...    | ...      | ...     | ...       | ✅/❌   |

    ### Regressions Detected
    (if any)

    <details>
    <summary>Degraded examples (N of M)</summary>
    | Input | Expected | Got |
    </details>
    """
    ...
```

### 2.4 GitHub Integration

```python
# src/scaffold/integrations/github.py

def post_pr_comment(report_md: str, github_token: str, repo: str, pr_number: int):
    """Post or update a PR comment with the eval report.

    Uses the GitHub API. Identifies existing Scaffold comments by a hidden
    marker and updates in place (avoids duplicate comments on re-runs).
    """
    ...

def detect_github_context() -> dict | None:
    """
    Read GitHub Actions environment variables:
    - GITHUB_REPOSITORY
    - GITHUB_EVENT_PATH (contains PR number)
    - GITHUB_TOKEN
    Returns None if not running in GitHub Actions.
    """
    ...
```

### 2.5 GitHub Action

```yaml
# action.yml
name: "Scaffold Eval"
description: "Run Scaffold LLM evals and post results"
inputs:
  compare-to:
    description: "Branch to compare baselines against"
    default: "origin/main"
  github-token:
    description: "Token for posting PR comments"
    default: ${{ github.token }}
runs:
  using: "composite"
  steps:
    - run: pip install scaffold-ai
    - run: scaffold run --compare-to=${{ inputs.compare-to }} --output=scaffold-report.md
    - run: |
        # Post comment using GitHub API
        ...
```

### 2.6 Dogfood Workflow

In addition to the user-facing GitHub Action, add a **dogfood workflow** that runs Scaffold on its own example directories. This provides CI validation for every subsequent phase automatically — when Phase 3 adds an LLM judge example, the dogfood workflow picks it up on the next push.

```yaml
# .github/workflows/scaffold-dogfood.yml
name: Scaffold Dogfood
on: [pull_request, push]

jobs:
  dogfood:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        example: [01-ci-regression]       # grows as examples are added
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e .
      - run: scaffold run
        working-directory: examples/${{ matrix.example }}
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### 2.7 Phase 2 Acceptance (End-to-End)

- [ ] `scaffold run --update-baseline` saves baselines to `.scaffold/baselines/`
- [ ] `scaffold run --compare-to=main` loads baselines and detects regressions
- [ ] PR report shows baseline vs current with threshold status
- [ ] Degraded examples are listed when regressions are detected
- [ ] **Real GitHub Actions run on an actual PR** — not just local testing. This validates the full CI path: checkout, baseline loading via `git show`, PR comment posting, exit code gating.
- [ ] PR comment is posted and updated on re-runs (no duplicate comments)
- [ ] Exit code 1 when any threshold is violated
- [ ] Dogfood workflow runs `examples/01-ci-regression/` successfully in CI

> **From this point forward**, every subsequent phase is implicitly CI-validated. The dogfood workflow matrix is expanded as new examples are added (Phase 3 adds `03-llm-as-judge`, Phase 5 adds `02-model-migration`, etc.). No phase after Phase 2 needs a manual GitHub Actions acceptance step — the dogfood workflow catches regressions automatically.

---

## Phase 3: LLM-as-Judge + Custom Judges

**Goal:** Support open-ended evaluation tasks where deterministic metrics don't work.

### 3.1 LLM Judge

```python
# src/scaffold/judges/llm_judge.py

class LLMJudge(Judge):
    """Evaluates outputs using an LLM against a rubric."""

    def __init__(self, model: str, rubric: list[dict]):
        """
        model: litellm model string (e.g., "openai/gpt-4o")
        rubric: list of {"id": str, "prompt": str} criteria
        """
        self.model = model
        self.rubric = rubric

    async def evaluate_single(self, input: str, expected: str, actual: str) -> JudgeResult:
        """
        For each rubric criterion:
        1. Construct a judge prompt with the criterion, input, expected, and actual
        2. Call litellm.acompletion asking for a structured yes/no + reasoning
        3. Parse the response

        Score = fraction of criteria passed.
        Reason = summary of failed criteria.
        """
        ...

    async def evaluate_dataset(self, examples, results) -> dict[str, float]:
        """
        Run evaluate_single on all examples (with parallelism).
        Returns {"rubric_pass_rate": float} — fraction of examples where all criteria passed.
        Also returns per-criterion pass rates for the detail report.
        """
        ...
```

**Judge prompt template** (the most important design decision in this module):

```
You are an evaluation judge. Given a user input, a reference answer, and an actual model output,
evaluate whether the actual output satisfies the following criterion.

## Criterion
{criterion_prompt}

## User Input
{input}

## Reference Answer
{expected}

## Actual Output
{actual}

## Instructions
Respond with a JSON object:
{{"passed": true/false, "reasoning": "brief explanation"}}
```

**Design notes:**
- Structured JSON output avoids brittle parsing.
- Each criterion is evaluated independently (separate LLM call per criterion per example). This is more expensive but more reliable than evaluating all criteria in one call.
- Uses `temperature: 0` by default for maximum reproducibility.
- Result caching: hash(model + criterion + input + expected + actual) → cached result stored in `.scaffold/cache/judge_results.json`. Cache is invalidated when the rubric changes. Avoids re-running identical judge calls across runs.
- Patch target for mocking in tests: `scaffold.judges.llm_judge.litellm.acompletion` (import at module level so patch-at-use-site works).

**Acceptance criteria:**
- [ ] Evaluates each rubric criterion independently
- [ ] Handles LLM judge failures (timeout, malformed response) gracefully — counts as a criterion failure with reason
- [ ] Respects parallelism settings
- [ ] Caches results to `.scaffold/cache/` and skips identical calls on re-runs
- [ ] Cache hit rate reported in `--verbose` mode
- [ ] Works with any litellm-supported model

### 3.2 Custom Judge

```python
# src/scaffold/judges/custom.py
import importlib.util

class CustomJudge(Judge):
    """Loads and runs a user-defined Python judge function."""

    def __init__(self, module_path: str, function_name: str):
        self.fn = self._load_function(module_path, function_name)

    def _load_function(self, module_path: str, function_name: str):
        """
        Dynamically import a function from a Python file.
        Path is resolved relative to scaffold.yaml's directory.
        The function must have signature: (input: str, expected: str, actual: str) -> dict
        The dict must contain at minimum: {"score": float}  (0.0 to 1.0)
        Optional: {"score": float, "reason": str}
        """
        spec = importlib.util.spec_from_file_location("custom_judge", module_path)
        if spec is None or spec.loader is None:
            raise ConfigError(f"Cannot load module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, function_name, None)
        if fn is None:
            raise ConfigError(f"Function '{function_name}' not found in {module_path}")
        return fn

    async def evaluate_single(self, input, expected, actual) -> JudgeResult:
        result = self.fn(input, expected, actual)
        return JudgeResult(
            score=result["score"],
            reason=result.get("reason"),
        )
```

**Acceptance criteria:**
- [ ] Loads arbitrary Python files and functions by path
- [ ] Validates the return schema (must have "score")
- [ ] Handles import errors, missing functions, runtime errors in user code with clear messages
- [ ] User function receives raw strings, not Scaffold internals

### 3.3 Judge Factory

```python
# src/scaffold/judges/factory.py

def create_judge(config: JudgeConfig) -> Judge:
    """Route judge config to the correct implementation.
    All returned judges have async evaluate_single/evaluate_dataset methods."""
    match config.type:
        case "exact_match":
            return ExactMatchJudge()
        case "llm":
            if not config.model or not config.rubric:
                raise ConfigError("LLM judge requires 'model' and 'rubric' fields")
            return LLMJudge(model=config.model, rubric=config.rubric)
        case "custom":
            if not config.module or not config.function:
                raise ConfigError("Custom judge requires 'module' and 'function' fields")
            return CustomJudge(module_path=config.module, function_name=config.function)
        case "composite":
            # Phase 6 only — agent composite judges
            raise ConfigError("Composite judges require level: agent (Phase 6)")
        case _:
            raise ConfigError(f"Unknown judge type: {config.type}")
```

### 3.4 Phase 3 Acceptance

- [ ] Create `examples/03-llm-as-judge/` with rubric-based eval
- [ ] Create `examples/04-custom-judge/` with a JSON schema validator
- [ ] LLM judge produces consistent scores across runs (temperature handling)
- [ ] Custom judge loads and runs user Python without modifying Scaffold
- [ ] Result caching prevents redundant LLM calls on re-runs

---

## Phase 4: Advanced Dataset Tooling

**Goal:** `scaffold dataset check` (coverage analysis) and `scaffold dataset import` (CSV/JSON import). Basic `init` and `add` were shipped in Phase 1.

### 4.1 Dataset Check (Coverage Analysis)

This is the most valuable dataset command — it tells users *where* their dataset is weak.

```python
# src/scaffold/dataset/check.py

@dataclass
class CoverageReport:
    total_examples: int
    categories: dict[str, int]          # expected_value → count
    warnings: list[str]                 # underrepresented categories, etc.
    suggestions: list[str]              # "Add more examples for category X"

def check_dataset(name: str, base_dir: Path = Path("evals")) -> CoverageReport:
    """
    Analyze the dataset for:
    - Total count
    - Distribution across expected values (categories)
    - Underrepresented categories (< min_threshold, default 15)
    - Duplicate inputs
    - Input length distribution (flag outliers)
    - For agent datasets: constraint coverage, tool coverage
    """
    ...
```

### 4.4 Dataset Import

```python
# src/scaffold/dataset/import_data.py

def import_dataset(name: str, source_path: Path, base_dir: Path = Path("evals")):
    """
    Import from CSV or JSON into JSONL format.

    CSV: expects columns named 'input' and 'expected' (or configurable mapping).
    JSON: expects array of objects with 'input' and 'expected'.

    Validates each row, reports skipped rows with reasons.
    """
    ...
```

### 4.3 Phase 4 Acceptance

- [ ] `scaffold dataset check --name test` reports coverage gaps, duplicate inputs, and length outliers
- [ ] Coverage report flags underrepresented categories by name with suggested count
- [ ] `scaffold dataset import --name test --from data.csv` imports CSV data with column mapping
- [ ] `scaffold dataset import --name test --from data.json` imports JSON array
- [ ] Import reports skipped rows with reasons (missing fields, parse errors)

---

## Phase 5: Model Migration

**Goal:** Automated prompt re-tuning when switching models. This is Scaffold's strongest differentiator — no competitor offers this.

### 5.1 Data Splitter

```python
# src/scaffold/migrate/splitter.py

@dataclass
class DataSplit:
    train: list[EvalExample]
    validation: list[EvalExample]
    holdout: list[EvalExample]

def split_dataset(
    examples: list[EvalExample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    """
    Deterministic random split. Stratified by expected value to maintain
    category distribution across splits.
    """
    ...
```

### 5.2 Prompt Optimizer

```python
# src/scaffold/migrate/optimizer.py

@dataclass
class OptimizationStep:
    iteration: int
    prompt_text: str
    train_score: float
    val_score: float
    diff: str                           # unified diff from previous prompt

@dataclass
class OptimizationResult:
    best_prompt: str
    best_val_score: float
    holdout_score: float
    original_score: float               # baseline on old model
    steps: list[OptimizationStep]
    stopped_reason: str                 # "converged", "patience", "max_iterations"

OPTIMIZER_SYSTEM_PROMPT = """You are a prompt optimization assistant. You will be given:
1. The current prompt
2. The model it will run on
3. Examples where the prompt failed (input, expected output, actual output)
4. The current score and target score

Your job is to suggest a MINIMAL modification to the prompt that fixes the failures.

Rules:
- Change as little as possible. Do not rewrite from scratch.
- Prefer rewording existing instructions over adding new ones.
- Explain your reasoning in <reasoning> tags, then output the full modified prompt in <prompt> tags.
"""

async def optimize_prompt(
    original_prompt: str,
    from_model: str,
    to_model: str,
    optimizer_model: str,
    eval_config: EvalConfig,
    split: DataSplit,
    primary_metric: str,
    patience: int = 3,
    min_improvement: float = 0.005,
    max_iterations: int = 20,
    max_edit_distance: int | None = None,
) -> OptimizationResult:
    """
    The optimization loop:

    0. Evaluate original prompt on holdout set with FROM model → original_score
       (this is the target to match — the quality bar on the old model)
    1. Evaluate original prompt on train set with TO model → starting score on new model
    2. For each iteration:
       a. Identify failure examples from the train set
       b. Ask optimizer LLM to suggest a prompt modification
       c. Parse the new prompt from the response
       d. (Optional) Reject if edit distance exceeds max_edit_distance
       e. Evaluate new prompt on train set with TO model
       f. Evaluate new prompt on validation set with TO model
       g. If val_score improved: update best, reset patience counter
       h. If not improved: increment patience counter
       i. If patience exhausted or max_iterations reached: stop
    3. Evaluate best prompt on holdout set with TO model → final honest score
    4. Return OptimizationResult (includes original_score from step 0 for comparison)

    primary_metric: which metric from eval_config.metrics to optimize (e.g., "f1_macro").
    When eval has multiple metrics, the optimizer focuses on one; others are reported.
    """
    ...
```

### 5.3 Early Stopping

```python
# src/scaffold/migrate/stopping.py

class EarlyStopping:
    """Tracks validation scores and determines when to stop."""

    def __init__(self, patience: int = 3, min_improvement: float = 0.005):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_score: float | None = None
        self.stale_count = 0

    def should_stop(self, val_score: float) -> bool:
        if self.best_score is None or val_score > self.best_score + self.min_improvement:
            self.best_score = val_score
            self.stale_count = 0
            return False
        self.stale_count += 1
        return self.stale_count >= self.patience
```

### 5.4 Migration Report

```python
# src/scaffold/migrate/report.py

def format_migration_report(result: OptimizationResult) -> str:
    """
    ## Migration Report: {from_model} → {to_model}

    ### Optimization Summary
    - Iterations: N (stopped: reason)
    - Train score: X → Y
    - Validation score: X → Y
    - Holdout score: X (original: Z on old model)

    ### Prompt Diff
    (unified diff between original and optimized prompt)

    ### Iteration History
    | # | Train | Val | Change |
    |---|-------|-----|--------|
    | 1 | 0.85  | 0.84| Rewording of instruction... |

    ### Remaining Regressions
    (holdout examples that still fail)
    """
    ...
```

### 5.5 CLI Integration

Wire `scaffold migrate` in cli.py:

```python
@cli.command()
def migrate(from_model, to_model, eval_name, optimizer_model, patience, max_iterations, min_improvement, max_edit_distance):
    config = load_config()
    eval_cfg = find_eval(config, eval_name)
    examples = load_dataset(eval_cfg.dataset)
    split = split_dataset(examples)
    primary_metric = eval_cfg.metrics[0].name  # optimize the first listed metric

    result = asyncio.run(optimize_prompt(
        original_prompt=load_prompt(config.target.prompt_file),
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
    ))

    report = format_migration_report(result)
    click.echo(report)

    # Optionally write the optimized prompt
    if click.confirm("Write optimized prompt to disk?"):
        write_prompt(config.target.prompt_file, result.best_prompt)
```

### 5.6 Phase 5 Acceptance

- [ ] Create `examples/02-model-migration/` with a working migration example
- [ ] `scaffold migrate --from gpt-4o --to gpt-4.5 --eval ticket-classification` runs the loop
- [ ] Optimizer makes small, targeted changes (not full rewrites)
- [ ] Early stopping kicks in when improvement plateaus
- [ ] Holdout score is computed only at the end (not leaked during optimization)
- [ ] Prompt diff is clear and minimal
- [ ] Max edit distance rejection works when configured
- [ ] Migration report includes iteration history

---

## Phase 6: Agent Evaluation (v2)

**Goal:** Evaluate agentic workflows — single-turn and multi-turn — with composite judging over execution traces.

### 6.1 Agent Data Models

```python
# src/scaffold/models.py (extend existing)

class AgentConstraints(BaseModel):
    max_tool_calls: int | None = None
    required_tools: list[str] | None = None
    forbidden_tools: list[str] | None = None
    max_tokens: int | None = None

class AgentExpected(BaseModel):
    outcome: str
    constraints: AgentConstraints | None = None

class AgentTurn(BaseModel):
    user_message: str
    context: dict | None = None         # only on first turn
    expected: AgentExpected

class AgentScenario(BaseModel):
    """Single-turn or multi-turn agent eval example."""
    input: dict | None = None           # single-turn
    expected: AgentExpected | None = None  # single-turn
    turns: list[AgentTurn] | None = None  # multi-turn
    conversation_constraints: AgentConstraints | None = None

class TraceStep(BaseModel):
    step: int
    type: Literal["tool_call", "response"]
    tool: str | None = None
    args: dict | None = None
    content: str | None = None
    tokens: int | None = None

class AgentTrace(BaseModel):
    """Single-turn or multi-turn trace output."""
    final_output: str | None = None
    trace: list[TraceStep] | None = None
    total_tool_calls: int | None = None
    total_tokens: int | None = None
    turns: list[dict] | None = None     # multi-turn: list of per-turn traces
```

### 6.2 Agent Target Runner

```python
# src/scaffold/targets/agent.py

async def run_agent_target_single_turn(
    command_template: str,
    scenario: AgentScenario,
    timeout: int = 60,
) -> AgentTrace:
    """
    1. Write scenario input to temp input file
    2. Run command with {input_file} and {output_file} (for the trace)
    3. Parse the trace JSON from output file
    4. Return AgentTrace
    """
    ...

async def run_agent_target_multi_turn(
    command_template: str,
    scenario: AgentScenario,
    mode: str,       # "full_replay" or "history_injection"
    timeout: int = 60,
) -> AgentTrace:
    """
    full_replay:
      For each turn, invoke the command with the cumulative conversation
      history. Collect per-turn traces. Combine into a multi-turn AgentTrace.

    history_injection:
      Invoke the command once with all prior turns as pre-filled history.
      Only the final turn is actually executed.
    """
    ...
```

### 6.3 Composite Judge

```python
# src/scaffold/judges/composite.py

class ConstraintJudge:
    """Deterministic check against tool/token/latency budgets."""

    def evaluate(self, trace: AgentTrace, constraints: AgentConstraints) -> JudgeResult:
        """
        Check each constraint:
        - max_tool_calls: trace.total_tool_calls <= max
        - required_tools: all required tools appear in trace
        - forbidden_tools: no forbidden tools appear in trace
        - max_tokens: trace.total_tokens <= max

        Score = fraction of constraints passed.
        """
        ...

class TrajectoryJudge:
    """LLM-based evaluation of execution path quality."""

    async def evaluate(self, trace: AgentTrace, rubric: str) -> JudgeResult:
        """Ask an LLM to evaluate the trace against the trajectory rubric."""
        ...

class OutcomeJudge:
    """LLM-based evaluation of final outcome correctness."""

    async def evaluate(self, scenario: AgentScenario, trace: AgentTrace) -> JudgeResult:
        """Ask an LLM to evaluate the final output against expected outcome."""
        ...

class CompositeJudge(Judge):
    """Weighted combination of outcome, constraint, and trajectory judges."""

    def __init__(self, criteria: list[dict]):
        """
        criteria: list of {"name": str, "type": str, "weight": float, ...}
        Types: "llm" (outcome/trajectory), "constraint"
        """
        self.criteria = criteria

    async def evaluate_scenario(
        self, scenario: AgentScenario, trace: AgentTrace
    ) -> dict[str, JudgeResult]:
        """
        Run each criterion judge, return per-criterion results.
        Composite score = weighted sum of individual scores.
        """
        ...
```

### 6.4 Phase 6 Acceptance

- [ ] Create `examples/05-agent-single-turn/` with composite judging
- [ ] Create `examples/06-agent-multi-turn/` with conversation testing
- [ ] Composite judge correctly weights outcome, trajectory, and constraint scores
- [ ] Multi-turn full_replay mode calls command once per turn
- [ ] Multi-turn history_injection mode calls command once
- [ ] Constraint violations are detected deterministically (no LLM needed)
- [ ] Agent traces are validated against the expected schema

---

## Phase 7: Polish, Examples & Ecosystem

**Goal:** Production-ready release with documentation, examples, and migration paths.

### 7.1 `scaffold init`

Interactive config generator:

1. Detect prompt files in the current directory
2. Ask: command mode or direct API mode?
3. Ask: what type of task? (classification / open-ended / agent)
4. Generate a starter `scaffold.yaml`
5. Generate a starter dataset with 3-5 placeholder examples
6. Print next steps

### 7.2 Promptfoo Config Import

```python
# src/scaffold/import_promptfoo.py

def import_promptfoo_config(source: Path, output: Path = Path("scaffold.yaml")):
    """
    Parse a promptfooconfig.yaml and convert to scaffold.yaml.

    Mapping:
    - providers → target
    - prompts → prompt_file
    - tests[].assert → metrics with thresholds
    - tests[].vars → dataset rows

    Emit warnings for unsupported features (red teaming plugins, etc.).
    """
    ...
```

### 7.3 Runnable Examples

Complete the examples directory (one per use case from the use-cases page):

| Example | Demonstrates |
|---------|-------------|
| `01-ci-regression/` | Ticket classifier with exact_match + F1 |
| `02-model-migration/` | Migrate GPT-4o → GPT-4.5 |
| `03-llm-as-judge/` | Open-ended generation with rubric judging |
| `04-custom-judge/` | JSON schema validation with a Python judge |
| `05-agent-single-turn/` | Single-turn agent with composite judging |
| `06-agent-multi-turn/` | Multi-turn conversation testing |
| `07-pipeline-level/` | Full pipeline test (RAG + LLM) |

Each example includes a README with what it demonstrates, how to run it, and how to adapt it.

### 7.4 Framework Adapters (v2)

```python
# src/scaffold/integrations/openai_agents.py

def traced_agent(agent, trace_output: str):
    """
    Wrap an OpenAI Agent SDK agent to capture execution traces.

    Returns a wrapper that:
    1. Runs the agent normally
    2. Captures tool calls, responses, and token usage
    3. Writes a Scaffold-format trace JSON to trace_output

    Test-time only — never used in production.
    """
    ...
```

Similar implementations for `pydantic_ai.py` and `claude_agents.py`.

### 7.5 Phase 7 Acceptance

- [ ] `scaffold init` generates a working config from scratch
- [ ] `scaffold import-promptfoo` converts basic Promptfoo configs
- [ ] All 7 examples are runnable with just an API key
- [ ] README covers installation, quickstart, and each feature
- [ ] PyPI package `scaffold-ai` installable
- [ ] GitHub Action wrapper works end-to-end

---

## Cross-Cutting Concerns

### Error Handling Strategy

Every user-facing error should include:
1. **What went wrong** (the error)
2. **Where it happened** (file, line, eval name)
3. **How to fix it** (suggestion)

```
Error: Dataset 'evals/tickets.jsonl' line 47: missing required field 'expected'

  {"input": "My printer won't connect to wifi"}
                                                  ^ missing 'expected'

Fix: Add an 'expected' field to each line in your JSONL dataset.
```

Use `rich` for colored output. Errors are red, warnings are yellow, success is green.

### Async Architecture

- Target runners are async (IO-bound: subprocess calls, API calls).
- Judges are async where they make LLM calls (LLM judge), sync otherwise (exact match, metrics, custom).
- The runner uses `asyncio.Semaphore` for parallelism control.
- The CLI uses `asyncio.run()` as the entry point.

### Testing Strategy

#### Test Directory Structure

```
tests/
├── conftest.py                     # shared fixtures: sample configs, datasets, mock targets
├── fixtures/
│   ├── configs/
│   │   ├── valid_minimal.yaml      # simplest valid config
│   │   ├── valid_full.yaml         # every field populated
│   │   ├── invalid_no_target.yaml
│   │   ├── invalid_bad_judge.yaml
│   │   └── ...
│   ├── datasets/
│   │   ├── classification_10.jsonl  # small deterministic dataset
│   │   ├── open_ended_5.jsonl       # LLM-as-judge dataset
│   │   ├── malformed.jsonl          # bad lines for error handling tests
│   │   ├── agent_single_turn.jsonl
│   │   └── agent_multi_turn.jsonl
│   └── mock_scripts/
│       ├── echo_target.py           # returns input as output (for wiring tests)
│       ├── classify_target.py       # deterministic classifier (for metric tests)
│       ├── failing_target.py        # exits non-zero (for error handling tests)
│       ├── slow_target.py           # sleeps past timeout (for timeout tests)
│       └── agent_target.py          # outputs a trace JSON (for agent tests)
├── unit/
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_exact_match.py
│   ├── test_metrics.py
│   ├── test_comparison.py
│   ├── test_report.py
│   └── test_splitter.py
├── integration/
│   ├── test_command_target.py
│   ├── test_direct_target.py
│   ├── test_llm_judge.py
│   ├── test_custom_judge.py
│   ├── test_baseline.py
│   ├── test_runner.py
│   ├── test_migration.py
│   └── test_agent_runner.py
└── e2e/
    ├── test_cli_run.py              # Phase 1
    ├── test_cli_migrate.py          # Phase 5
    ├── test_cli_dataset.py          # Phase 1 (basic), Phase 4 (check/import)
    └── test_cli_init.py             # Phase 7
```

**Tests are added in the same phase as their feature.** The full tree above shows the end state. Agent fixtures (`agent_single_turn.jsonl`, `agent_target.py`) are not added until Phase 6. E2E tests for `migrate` are not added until Phase 5. The CI workflow runs all tests that exist at any given phase.

#### Test Levels

| Level | Scope | Speed | LLM calls | When to run |
|-------|-------|-------|-----------|-------------|
| **Unit** | Single function/class, no IO | < 1s each | Never | Every push (CI) |
| **Integration** | Multiple modules, file IO, subprocesses | 1-10s each | Mocked | Every push (CI) |
| **E2E** | Full CLI invocation on example dirs | 5-30s each | Mocked | Every push (CI) |
| **Live** | Full CLI with real LLM API calls | 30-120s each | Real | Manual / nightly, never on PR CI |

#### What to Test Per Module

| Module | Unit tests | Integration tests |
|--------|-----------|-------------------|
| **Config parser** | Valid/invalid YAML, normalization (string judge → JudgeConfig), missing fields, unknown fields, type coercion | Config + dataset loader together (paths resolve correctly) |
| **Dataset loader** | Valid JSONL parsing, malformed lines (line number in error), empty files, extra fields preserved, smoke sampling determinism | Load from disk, encoding edge cases (UTF-8 BOM) |
| **ExactMatchJudge** | Exact match with stripping, case sensitivity flag, empty strings, whitespace-only | — |
| **MetricsJudge** | F1 macro/micro/weighted against hand-computed values, single-class edge case, all-wrong predictions, empty predictions | Verify against sklearn on a reference dataset (test dependency, not runtime) |
| **LLM Judge** | Prompt construction, response parsing (valid JSON, malformed JSON, missing fields), score aggregation | Full judge with mocked litellm (deterministic canned responses) |
| **Custom Judge** | — | Load real Python file, call function, validate return schema, handle import errors, handle runtime errors |
| **Command target** | — | Run `echo_target.py`, verify input/output file plumbing. Run `failing_target.py`, verify error handling. Run `slow_target.py`, verify timeout. |
| **Direct target** | — | Mock `litellm.acompletion`, verify provider/model routing, retry logic |
| **Baseline storage** | JSON serialization/deserialization, missing baseline handling | Save/load cycle on disk. Mock `git show` for cross-branch loading. |
| **Comparison** | Absolute threshold math, relative threshold math (percentage drop), division-by-zero (baseline=0), missing baseline skip | — |
| **Report** | Markdown table formatting, degraded examples rendering, pass/fail summary | Full report from EvalResult + ThresholdResult fixtures |
| **Runner** | — | End-to-end: config → dataset → mock target → real judge → report. Verify metrics, exit code. |
| **Migration** | Early stopping logic (patience counter, min improvement), data splitting (stratification, ratios, determinism) | Full optimization loop with mock optimizer LLM (returns predictable prompt edits) |
| **Agent runner** | Trace parsing, constraint checking (required/forbidden tools, token budget) | Single-turn and multi-turn with `agent_target.py`. Full replay vs history injection modes. |
| **Composite judge** | Weighted score calculation, per-criterion aggregation | Full composite with mocked LLM judges + real constraint judge |

#### Mocking LLM Calls

Real LLM calls are banned from CI tests. Three mocking strategies, used depending on context:

**1. Patching litellm.acompletion (most common)**

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

class MockResponse:
    """Mimics litellm's completion response structure."""
    def __init__(self, content: str):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.usage = MagicMock(total_tokens=100, prompt_tokens=80, completion_tokens=20)

@pytest.fixture
def mock_llm():
    """Fixture that patches litellm at the USE SITE, not the import site."""
    async def canned_response(*args, **kwargs):
        return MockResponse(content='{"passed": true, "reasoning": "Looks good"}')

    # Patch where litellm is used, not where it's defined
    with patch("scaffold.judges.llm_judge.litellm.acompletion",
               new=AsyncMock(side_effect=canned_response)) as mock:
        yield mock
```

Use this for: LLM judge tests, direct target tests, migration optimizer tests. Always patch at the use site (e.g., `scaffold.judges.llm_judge.litellm.acompletion`) not at the import site (`litellm.acompletion`).

**2. Deterministic mock scripts (for command mode)**

```python
# tests/fixtures/mock_scripts/classify_target.py
"""Deterministic classifier for testing. Maps known inputs to known outputs."""
import json, sys

RULES = {
    "My printer won't connect": "hardware",
    "I need a refund": "billing",
}

input_data = json.load(open(sys.argv[1]))
output = RULES.get(input_data["input"], "general")
json.dump({"output": output}, open(sys.argv[2], "w"))
```

Use this for: command target tests, runner integration tests, baseline comparison tests. The mock script is deterministic, so test assertions are exact.

**3. Response fixtures (for complex LLM interactions)**

```python
# tests/fixtures/llm_responses/
# migration_step_1.json — optimizer suggests a prompt change
# migration_step_2.json — optimizer suggests another change
# migration_step_3.json — optimizer says no more changes needed

@pytest.fixture
def mock_optimizer_llm(request):
    """Steps through a sequence of canned optimizer responses."""
    responses = load_fixture_sequence("llm_responses/migration_*.json")
    call_count = 0

    async def sequential_response(*args, **kwargs):
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return MockResponse(content=resp)

    with patch("litellm.acompletion", new=AsyncMock(side_effect=sequential_response)) as mock:
        yield mock
```

Use this for: migration tests (multi-step optimization loop), multi-turn agent tests.

#### Handling Non-Determinism

LLM outputs are inherently non-deterministic. This affects two areas:

**In Scaffold's own tests (our CI):** fully mocked, so determinism is guaranteed. No issue.

**In the user's evals (Scaffold's purpose):** Scaffold should help users handle this, but it's a runtime concern, not a test concern for us. Relevant design decisions:
- LLM judge uses `temperature: 0` by default for maximum reproducibility.
- `scaffold run` accepts `--seed` for reproducible sampling.
- Smoke test sampling uses a fixed seed.
- Migration holdout split uses a fixed seed.

#### Test CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI
on: [pull_request, push]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: mypy src/

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ matrix.python-version }}" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ tests/integration/ -v --tb=short
      - run: pytest tests/e2e/ -v --tb=short
```

The dogfood workflow (from Phase 2) is separate and uses real API keys. The CI workflow above runs on every push with zero external dependencies.

#### Coverage Targets

Not a hard gate initially, but track it:
- **Unit tests:** aim for >90% line coverage on config, dataset, judges, comparison, report modules.
- **Integration tests:** aim for >80% on target runners, baseline, migration.
- **E2E tests:** cover every CLI command with at least one happy path and one error path.

Use `pytest-cov` and print a summary in CI. Don't block PRs on coverage initially — it creates perverse incentives to write low-value tests. Revisit after v1 launch.

### Logging

- Default: only report output (clean for CI).
- `--verbose` / `-v`: show progress (which eval is running, how many examples done).
- `--debug`: full logging (LLM prompts/responses, subprocess commands, timing).
- Use Python's `logging` module, configured via the CLI.

---

## Dependency Graph

```
Phase 0 ─── Phase 1 ─── Phase 2 ─── Phase 3
  (repo)     (eval+ds)   (CI)        (LLM+custom judges)
                                        │
              Phase 4 ◄─────────────────┤  (can run in parallel with Phase 3)
              (adv dataset)             │
                                        │
              Phase 5 ◄─────────────────┘
              (migration)
                │
              Phase 6
              (agents + composite judges)
                │
              Phase 7
              (polish + examples)
```

- Phase 1 includes basic `scaffold dataset init/add` (per PLAN build sequence).
- Phase 4 (advanced dataset tooling: `check`, `import`) is independent of Phase 3 and can run in parallel.
- Phase 5 (migration) depends on Phase 3 (needs LLM judge for open-ended eval during optimization).
- Phase 6 (agents) introduces composite judges, constraint judges, and trajectory judges — these are new judge types, not from Phase 3. Phase 6 depends on Phase 5 for agent migration.
- Phase 7 (polish) depends on all prior phases for examples and documentation.

---

## Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 0: Scaffolding | 1 day | Day 1 |
| 1: Core eval loop + basic dataset CLI | 5-7 days | Week 1 |
| 2: Baselines + CI | 3-4 days | Week 2 |
| 3: LLM + custom judges | 3-4 days | Week 2-3 |
| 4: Advanced dataset tooling | 2-3 days | Week 3 |
| 5: Migration | 5-7 days | Week 4 |
| 6: Agent eval | 5-7 days | Week 5 |
| 7: Polish + examples | 3-5 days | Week 6 |

**First usable version (Phases 0-2): ~2 weeks.** This is enough for a team to add CI-gated deterministic evals.

**Full v1 (Phases 0-5): ~4 weeks.** Includes LLM judging, dataset tooling, and migration.

**Full v2 (Phases 0-7): ~6 weeks.** Includes agent eval and polish.

These estimates assume one developer working full-time. Phases 3+4 can be parallelized.
