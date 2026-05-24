# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-05-24

### Added
- **`scaffold.trace.TraceBuilder`** — fluent helper to build agent eval output JSON (`final_output`, `trace`, tool counts)
- **`scaffold.integrations.openai_agents`** — convert OpenAI Agents SDK `RunResult` to Scaffold format; `run_for_scaffold` / `run_for_scaffold_sync`
- Optional extra: `pip install 'llmci[agents]'` (`openai-agents`)
- Example [`10-agent-openai-agents`](examples/10-agent-openai-agents/) — mock (CI) and real SDK paths

## [0.1.1] - 2026-05-24

### Fixed
- **Merged PR comments for CI matrix jobs** — parallel jobs no longer overwrite each other's eval reports. Set `SCAFFOLD_REPORT_SLICE` (e.g. `ticket-classifier/scaffold.yaml`) so each job updates its own slice in one combined comment.

## [0.1.0] - 2026-05-23

Initial public release on PyPI as [`llmci`](https://pypi.org/project/llmci/). The CLI command is `scaffold`.

### Added

**Core eval loop**
- `scaffold run` — run eval datasets, compute metrics, enforce thresholds, exit non-zero on failure
- `scaffold.yaml` config: command and direct (litellm) target modes, per-eval overrides
- Absolute and `max_regression` threshold modes with baseline storage in `.scaffold/baselines/`
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
- `scaffold migrate` — prompt optimization when switching models
- Train/validation/holdout splitting with early stopping

**Dataset tooling**
- `scaffold dataset init`, `add`, `check`, and `import` (CSV/JSON)
- `scaffold init` — interactive project setup
- `scaffold import-promptfoo` — convert Promptfoo configs

**CI integration**
- GitHub Actions composite action (`action.yml`)
- Automatic PR comments with eval report, regressions, and failed examples
- Direct API `base_url` support for internal LLM proxies

**Examples**
- Nine runnable examples: CI regression, model migration, LLM judge, custom judge, agent single/multi-turn, RAG pipeline, FastAPI service, summarization QA

[0.1.2]: https://github.com/alexminnaar/scaffold/releases/tag/v0.1.2
[0.1.1]: https://github.com/alexminnaar/scaffold/releases/tag/v0.1.1
[0.1.0]: https://github.com/alexminnaar/scaffold/releases/tag/v0.1.0
