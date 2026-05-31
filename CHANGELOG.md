# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[0.1.7]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.7
[0.1.6]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.6
[0.1.5]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.5
[0.1.4]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.4
[0.1.3]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.3
[0.1.2]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.2
[0.1.1]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.1
[0.1.0]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.0
