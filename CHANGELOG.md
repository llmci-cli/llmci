# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-06

Post-0.2.0 follow-ups: judge caching, calibration trends, gate trust warnings,
per-claim RAG faithfulness, LLM red-team mutation, and multimodal direct targets.

### Added
- **Multimodal direct targets** — dataset rows can include `images` and/or `audio`
  fields (paths relative to the dataset file, or HTTPS/data URLs). Direct API targets
  build litellm multimodal messages and include media in the response-cache key.
  Example: `examples/18-multimodal-vision`.
- **LLM attack mutation** — `llmci redteam generate --mutate` appends LLM-rephrased
  variants of each built-in attack (`--mutate-model`, `--mutate-count`, `--no-cache`).
  Mutated rows carry `parent_attack` and `mutated: true` metadata.
- **Per-claim faithfulness** — RAG `faithfulness` criteria accept
  `decompose_claims: true` to extract atomic claims and score each against context
  (fraction supported), with unsupported claims listed in the judge reason.
- **Gate configuration warnings** — `llmci run` warns when `max_regression` thresholds
  or a pairwise judge have no baseline, or when multi-sample runs lack `significance`.
- **Composite judge caching** — outcome and trajectory criteria in agent composite judges
  now share the `.llmci/cache/judges/` LLM-call cache (same `--no-cache` /
  `--refresh-cache` flags as other judges).
- **Calibration trend history** — `--save-snapshot` appends each run to
  `.llmci/calibration/<eval>-history.jsonl`; `llmci judge calibrate` shows a trend
  table when two or more calibration runs exist.

### Fixed
- Calibration timestamps use `timezone.utc` for Python 3.10 mypy compatibility.

## [0.2.0] - 2026-06-06

Major release: CI gate trust (flake resistance, caching, cost metrics, portable reports),
deeper eval quality (RAG, pairwise, calibration, diffs), safety/red-team, plugin API, and
seventeen runnable examples including an integrated pre-merge gate.

### Added
- **Integrated CI gate example** (`examples/17-integrated-ci-gate`) — one config that
  stacks quality (`accuracy`), cost/token regression vs committed baselines
  (`cost_mean`, `tokens_in_mean`), and safety (`pii_leakage`) in a single pre-merge
  gate. Fully deterministic, API-key-free, with committed baselines and failure toggles
  for each concern.
- **Auto-load local baselines** — when `--compare-to` is omitted, `llmci run` loads
  baselines from `.llmci/baselines/` on disk so checked-in baselines work without a git
  ref (PR flows still use `--compare-to=origin/main`).
- **LLM judge-call caching** — the `pairwise`, `rag`, and `safety` judges now cache their
  scoring calls under `.llmci/cache/judges/` (keyed on model + prompt), via a shared helper
  that honors the same `--no-cache` / `--refresh-cache` flags as target caching. This
  offsets the extra calls from RAG's multiple criteria and the pairwise position-swap;
  caching is skipped while sampling (`samples_per_example > 1`) so variance isn't flattened.
- **Per-criterion judge calibration** — `llmci judge calibrate` now calibrates each
  criterion of multi-criterion judges (composite / RAG / safety) against per-criterion
  human labels. Add a `criteria` dict to each labeled row (the overall score is derived as
  the mean when `human_score` is omitted); the report gains a per-criterion agreement
  table and `--min-agreement` fails if *any* criterion falls below the threshold.
- **Pairwise position-bias control** — the pairwise judge now runs each comparison in
  both A/B orders and averages them by default (`position_swap: true`), cancelling the
  LLM's tendency to favor a fixed position; a judge that blindly prefers one slot scores a
  neutral 0.5 instead of a false win. Set `position_swap: false` to halve judge calls.
- **Structured-output judge** (`judge: {type: structured}`) — validate a target's JSON
  output against a JSON Schema, gateable by name (1.0 valid / 0.0 invalid); no API key.
  The schema is inline under `json_schema:` or a path to a `.json` file. A self-contained
  validator covers the practical subset (`type`/`required`/`properties`/`items`/`enum`/
  `additionalProperties`/`minimum`/`maximum`/`minLength`/`maxLength`/`minItems`/`maxItems`/
  `pattern`). `partial_credit: true` scores the fraction of required fields that validate.
  See `examples/16-structured-output`.
- **Red-team attack generator** — `llmci redteam generate --seeds <file>` expands a few
  plain seed intents into many adversarially-framed prompts (jailbreak, prompt-injection,
  PII-extraction, and obfuscation techniques) for the `safety` judge to gate. Fully
  deterministic and API-key-free, so the generated dataset is reproducible and diffable in
  CI. Filter with `--category` / `--attack`, list the library with `--list`, and add
  `--include-control` for a raw-seed baseline. Each row carries `attack`/`category`/`seed`
  metadata so failures attribute to a specific technique. See `examples/15-redteam`.
- **Custom report sinks** — register a `(ReportContext) -> None` callable with
  `register_reporter` to ship results after each run (Slack, dashboards, artifact
  uploads). Activate via `reporters:` in `llmci.yaml`; sinks load from local modules
  (`plugins:`) or the `llmci.reporters` entry-point group. A sink that raises only warns
  and never changes the pass/fail gate. The `ReportContext` carries the eval results,
  configs, overall `passed` flag, and rendered markdown.
- **Plugin / extension API for judges and metrics** — register a custom `judge.type` or a
  custom metric (gateable by name) without forking. Installed packages advertise them via
  the `llmci.judges` / `llmci.metrics` entry-point groups; local repos list dotted module
  paths under `plugins:` in `llmci.yaml`. A judge value is a `Judge` subclass or a
  `(JudgeConfig) -> Judge` factory; a metric value is a `(MetricContext) -> float` callable
  (`register_metric(..., lower_is_better=True)` flips threshold direction). Plugin names
  can't shadow built-ins. `JudgeConfig.type` is now an open string validated at
  judge-creation time.
- **Safety / red-team judge** (`judge: {type: safety}`) — gate on `pii_leakage`
  (deterministic, no API key: scans for emails, phones, SSNs, credit cards, IPs, and
  AWS keys; `categories` narrows the scan), plus LLM-based `toxicity` and
  `jailbreak_resistance` criteria. Each criterion is a gateable metric by name where
  higher = safer (e.g. `{name: pii_leakage, threshold: 1.0, mode: absolute}`).
- **Judge calibration & drift detection** — `llmci judge calibrate --eval <name>
  --labels <file>` runs a configured judge over a human-labeled set and reports
  judge↔human agreement (agreement rate, Cohen's kappa, MAE, Pearson r). A calibration
  snapshot (`.llmci/calibration/<eval>.json`, written with `--save-snapshot`) records
  the judge model and scores so a later run flags drift when the judge model changes.
  Gate with `--min-agreement` and/or `--max-drift`.
- **Machine-readable & shareable report formats** — `llmci run --output-format
  junit|sarif|json|html` for CI systems beyond GitHub Actions. `html` is a
  self-contained, shareable run report (summary, regressions, per-example results).
  PR comments stay markdown. New `output-format` input on the GitHub Action.
- **Response caching** for direct API targets, keyed on
  `(provider, model, prompt, input)` under `.llmci/cache/responses/`. New flags
  `--no-cache` and `--refresh-cache`.
- **Flake resistance** — `settings.samples_per_example` (and `--samples`) run each
  eval over multiple rounds, reporting each metric's mean with a confidence interval.
  `settings.significance` (and `--significance`) gates `max_regression` thresholds so
  drops within run-to-run noise are reported but not enforced.
- **Cost / token budgeting** — new `cost_total`, `cost_mean`, `tokens_in_mean`,
  `tokens_out_mean`, and `tokens_total_mean` metrics. Cost/usage come from the litellm
  response for direct targets, or from optional `"usage"`/`"cost"` keys in a command
  target's output JSON.
- **Pairwise / preference judge** (`judge: {type: pairwise}`) — compares each current
  output against the baseline output for the same input and reports a `win_rate` metric
  (1.0 win / 0.5 tie / 0.0 loss). Optional `rubric` sets the comparison criterion.
  Pairs with `samples_per_example` for a win-rate confidence interval.
- **Output diffs vs baseline** — baselines now store per-example outputs, and reports
  (markdown + HTML) show a baseline-vs-current diff for each regressed example, matched
  by input. Backward compatible with baselines written before this change.
- **RAG judge** (`judge: {type: rag}`) — `faithfulness`, `answer_relevance`,
  `context_relevance`, `retrieval_recall`, and `retrieval_precision` criteria. Each
  becomes a gateable metric by name. Command targets pass retrieval context via
  `"contexts"`/`"retrieved_ids"` output keys; gold labels come from a dataset row's
  `relevant_ids`. Per-example judge sub-scores are now exposed as aggregate metrics.

### Fixed
- Local judge plugins listed under `plugins:` now resolve without packaging: the config
  directory is placed on `sys.path` while plugin modules are imported.
- Lower-is-better metrics (latency, cost, tokens, `error_rate`) now compare correctly:
  `absolute` thresholds require the value to be `<=` the threshold, and a
  `max_regression` is an increase rather than a drop.

## [0.1.9] - 2026-05-31

### Added
- Release metadata consistency check for package version, action install version, and changelog links.
- Manual real-LLM example workflow for API-key-dependent examples.
- GitHub Action inputs for explicit config paths, discovered config runs, and baseline updates.

### Fixed
- Duplicate llmci PR comments from parallel matrix jobs are merged into one canonical comment and stale duplicates are cleaned up.

## [0.1.8] - 2026-05-31

### Added
- `--include` and `--exclude` filters for `llmci discover` and `llmci run --all`.

## [0.1.7] - 2026-05-31

### Added
- `llmci discover` to list config files in a repository.
- `llmci run --all` to run every discovered config.

## [0.1.6] - 2026-05-31

### Added
- `llmci run --config <path>` to run evals from an alternate config file.

## [0.1.5] - 2026-05-25

### Fixed
- S3 dataset URI validation runs before the optional `boto3` import.
- PyPI publish workflow grants `contents: read` so checkout works alongside trusted publishing.

## [0.1.4] - 2026-05-25

### Added
- **Remote eval datasets** — `dataset` accepts `s3://` and `https://` URIs (string or `{source, cache}`). S3 requires `pip install 'llmci[s3]'`. Cached under `.llmci/cache/datasets/` by default.

### Changed
- Repository and package metadata URLs updated for the `llmci-cli` GitHub organization.

## [0.1.3] - 2026-05-24

### Changed
- **Full rebrand to `llmci`** — CLI command is now `llmci` (was `scaffold`); config file is `llmci.yaml` (was `scaffold.yaml`); data directory is `.llmci/` (was `.scaffold/`); matrix PR slice env var is `LLMCI_REPORT_SLICE` (was `SCAFFOLD_REPORT_SLICE`)
- Python package module renamed from `scaffold` to `llmci`
- OpenAI Agents helpers renamed: `run_for_llmci_sync`, `llmci_input_to_agent_input`, etc.

## [0.1.2] - 2026-05-24

### Added
- **`llmci.trace.TraceBuilder`** — fluent helper to build agent eval output JSON (`final_output`, `trace`, tool counts)
- **`llmci.integrations.openai_agents`** — convert OpenAI Agents SDK `RunResult` to llmci format; `run_for_llmci` / `run_for_llmci_sync`
- Optional extra: `pip install 'llmci[agents]'` (`openai-agents`)
- Example [`10-agent-openai-agents`](examples/10-agent-openai-agents/) — mock (CI) and real SDK paths

## [0.1.1] - 2026-05-24

### Fixed
- **Merged PR comments for CI matrix jobs** — parallel jobs no longer overwrite each other's eval reports. Set `LLMCI_REPORT_SLICE` (e.g. `ticket-classifier/llmci.yaml`) so each job updates its own slice in one combined comment.

## [0.1.0] - 2026-05-23

Initial public release on PyPI as [`llmci`](https://pypi.org/project/llmci/).

### Added

**Core eval loop**
- `llmci run` — run eval datasets, compute metrics, enforce thresholds, exit non-zero on failure
- `llmci.yaml` config: command and direct (litellm) target modes, per-eval overrides
- Absolute and `max_regression` threshold modes with baseline storage in `.llmci/baselines/`
- `--compare-to`, `--update-baseline`, `--smoke`, and `--output` flags
- Parallel execution with configurable timeouts and retries

**Judges**
- `exact_match` for classification and deterministic outputs
- LLM-as-judge with rubric criteria
- Custom Python judges (`module` + `function`)
- Composite judges for agents: constraint, outcome, and trajectory criteria

**Metrics (21)**
- Score: `accuracy`, `pass_rate`, `mean_score`, `median_score`, `min_score`, `max_score`, `error_rate`
- Classification: `f1_macro`, `f1_micro`, `f1_weighted`, `precision_*`, `recall_*` variants
- Similarity: `cosine_similarity`
- Latency: `latency_mean`, `latency_p50`, `latency_p90`, `latency_p99`

**Agent evaluation**
- Single-turn and multi-turn conversation datasets
- Full replay and history-injection modes
- Tool-call constraint checking (required/forbidden tools, budgets, token limits)

**Model migration**
- `llmci migrate` — prompt optimization when switching models
- Train/validation/holdout splitting with early stopping

**Dataset tooling**
- `llmci dataset init`, `add`, `check`, and `import` (CSV/JSON)
- `llmci init` — interactive project setup
- `llmci import-promptfoo` — convert Promptfoo configs

**CI integration**
- GitHub Actions composite action (`action.yml`)
- Automatic PR comments with eval report, regressions, and failed examples
- Direct API `base_url` support for internal LLM proxies

**Examples**
- Nine runnable examples: CI regression, model migration, LLM judge, custom judge, agent single/multi-turn, RAG pipeline, FastAPI service, summarization QA

[0.3.0]: https://github.com/llmci-cli/llmci/releases/tag/v0.3.0
[0.2.0]: https://github.com/llmci-cli/llmci/releases/tag/v0.2.0
[0.1.9]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.9
[0.1.8]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.8
[0.1.7]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.7
[0.1.6]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.6
[0.1.5]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.5
[0.1.4]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.4
[0.1.3]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.3
[0.1.2]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.2
[0.1.1]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.1
[0.1.0]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.0
