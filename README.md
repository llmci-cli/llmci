# llmci

CI-native regression testing and migration for LLMs.

Catch quality drops before they merge. Migrate models without breaking things.

llmci is not an observability tool — it's a **pre-merge safety gate**. Define eval datasets, set quality thresholds, and let CI block bad changes to your prompts, models, or pipelines.

## Installation

```bash
pip install llmci
```

Requires Python 3.10+.

## Quick Start

### 1. Initialize

```bash
llmci init
```

This creates a `llmci.yaml` config and a starter eval dataset. You'll be asked:
- **Target mode** — `command` (run any script) or `direct` (call an LLM API)
- **Task type** — classification, open-ended, or agent
- **Eval name** — what to call this eval

### 2. Define your eval dataset

Edit the generated `evals/<name>.jsonl`. Each line is a JSON object:

```json
{"input": "My printer won't connect to wifi", "expected": "hardware"}
{"input": "I need a refund for order #882", "expected": "billing"}
```

Or add examples interactively:

```bash
llmci dataset add --name my-eval
```

### 3. Run

```bash
llmci run
```

Output:

```
## llmci Eval Report

| Eval | Metric | Score | Threshold | Status |
|------|--------|-------|-----------|--------|
| ticket-classification | accuracy | 0.950 | ≥ 0.9 | ✅ |
| ticket-classification | f1_macro | 0.940 | ≥ 0.85 | ✅ |
```

Exit code 0 = all thresholds pass. Exit code 1 = regression detected.

## Configuration

`llmci.yaml` defines your target, evals, and settings:

```yaml
version: 1

target:
  command: "python3 run_prompt.py --input {input_file} --output {output_file}"

evals:
  - name: ticket-classification
    dataset: ./evals/tickets.jsonl
    judge: exact_match
    metrics:
      - name: accuracy
        threshold: 0.90
        mode: absolute
      - name: f1_macro
        threshold: 0.85
        mode: absolute

settings:
  parallelism: 5
  timeout_per_call: 30
  retries: 1
```

Use `--config` when your eval config has a different name or lives in a service directory:

```bash
llmci run --config llmci-prompt-level.yaml
```

### Target Modes

**Command mode** — wrap any script, any language:

```yaml
target:
  command: "python3 my_pipeline.py --input {input_file} --output {output_file}"
```

Your script reads a JSON input file and writes a JSON output file with an `"output"` key.

**Direct API mode** — call an LLM provider directly:

```yaml
target:
  direct:
    provider: openai
    model: gpt-4o-mini
  prompt_file: prompt.txt
```

Uses [litellm](https://github.com/BerriAI/litellm) under the hood, so any provider works (OpenAI, Anthropic, Azure, etc.). Set credentials via environment variables.

For internal proxies or custom gateways, add `base_url`:

```yaml
target:
  direct:
    provider: openai
    model: gpt-4o
    base_url: https://llm-proxy.internal.company.com/v1
  prompt_file: prompt.txt
```

### Judges

| Type | Use case | Config |
|------|----------|--------|
| `exact_match` | Classification, deterministic outputs | `judge: exact_match` |
| `llm` | Open-ended generation, summarization | `judge: {type: llm, model: gpt-4o, rubric: [...]}` |
| `custom` | Domain-specific logic (JSON validation, etc.) | `judge: {type: custom, module: ./judge.py, function: evaluate}` |
| `composite` | Agent evaluation with multiple criteria | `judge: {type: composite, criteria: [...]}` |

### Metrics

**Score-based:**
- `accuracy` — fraction of exact matches (score = 1.0)
- `pass_rate` — fraction of examples scoring >= 0.5
- `mean_score` — average judge score
- `median_score` — median judge score (robust to outliers)
- `min_score` / `max_score` — worst and best scores in dataset
- `error_rate` — fraction of examples that errored

**Classification:**
- `f1_macro`, `f1_micro`, `f1_weighted` — F1 score variants
- `precision_macro`, `precision_micro`, `precision_weighted` — precision variants
- `recall_macro`, `recall_micro`, `recall_weighted` — recall variants

**Similarity:**
- `cosine_similarity` — token-overlap cosine similarity between expected and actual

**Latency:**
- `latency_mean`, `latency_p50`, `latency_p90`, `latency_p99` — response time percentiles (ms)

Each metric supports two threshold modes:
- `absolute` — score must be >= threshold (for latency metrics, must be <= threshold)
- `max_regression` — drop from baseline must be <= threshold (e.g., 0.05 = max 5% drop)

## CI Integration

### GitHub Actions

Add to your workflow:

```yaml
- uses: llmci-cli/llmci@main
  with:
    compare-to: origin/main
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Or use the CLI directly:

```yaml
- run: pip install llmci
- run: llmci run --compare-to=origin/main
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

For monorepos, pass the service config explicitly:

```yaml
- run: llmci run --config services/api/llmci.yaml --compare-to=origin/main
```

When running in GitHub Actions, llmci automatically posts eval results as a PR comment.

For **matrix CI** (multiple services in parallel), set a unique slice per job so reports merge into one comment:

```yaml
env:
  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  LLMCI_REPORT_SLICE: ${{ matrix.service }}/${{ matrix.config }}
```

### Baselines

Store baseline scores on your main branch:

```bash
llmci run --update-baseline
```

Then compare PRs against that baseline:

```bash
llmci run --compare-to=main
```

## Model Migration

When switching models (e.g., GPT-4o to GPT-4.5), llmci can automatically tune your prompt to maintain quality parity:

```bash
llmci migrate \
  --from gpt-4o \
  --to gpt-4.5 \
  --eval ticket-classification \
  --optimizer-model gpt-4o
```

The optimizer:
1. Splits your dataset into train/validation/holdout
2. Iteratively suggests minimal prompt modifications
3. Stops when improvement plateaus (early stopping)
4. Reports the final holdout score vs. the original model

## Agent Evaluation

Test tool-using and conversational agents with composite judging:

```yaml
evals:
  - name: agent-tool-use
    level: agent
    dataset: ./evals/scenarios.jsonl
    judge:
      type: composite
      criteria:
        - name: constraints
          type: constraint
          weight: 1.0
        - name: outcome
          type: outcome
          weight: 2.0
```

Your agent runs as a **command** that reads llmci input JSON and writes trace JSON. Use `llmci.trace.TraceBuilder` to build output, or `llmci.integrations.openai_agents` for the OpenAI Agents SDK — see [`examples/10-agent-openai-agents`](examples/10-agent-openai-agents/).

Supports:
- **Single-turn** and **multi-turn** conversations
- **Constraint checking** — tool call budgets, required/forbidden tools, token limits
- **Outcome judging** — LLM-based evaluation of final output
- **Trajectory judging** — LLM-based evaluation of execution path quality
- **Full replay** or **history injection** modes for multi-turn

## Dataset Tools

```bash
# Initialize a new dataset
llmci dataset init --name my-eval --type classification

# Add examples interactively
llmci dataset add --name my-eval

# Analyze coverage and quality
llmci dataset check --name my-eval

# Import from CSV or JSON
llmci dataset import --name my-eval --from data.csv
```

## Migrating from Promptfoo

```bash
llmci import-promptfoo promptfooconfig.yaml
```

Converts providers, test assertions, and variables into llmci's format.

## Reference integration

The [`llmci-testbed`](https://github.com/llmci-cli/llmci-testbed) repository is a realistic customer monorepo that dogfoods `llmci` against full HTTP services, RAG pipelines, agents, and migration workflows. Each service maps to a docs case study and runs in GitHub Actions with mock LLM mode (no API cost on PRs).

| Testbed path | Case study |
|--------------|------------|
| [`services/ticket-classifier`](https://github.com/llmci-cli/llmci-testbed/tree/main/services/ticket-classifier) | FastAPI service |
| [`services/rag-qa`](https://github.com/llmci-cli/llmci-testbed/tree/main/services/rag-qa) | RAG pipeline |
| [`services/summarizer`](https://github.com/llmci-cli/llmci-testbed/tree/main/services/summarizer) | Summarization QA |
| [`services/support-agent`](https://github.com/llmci-cli/llmci-testbed/tree/main/services/support-agent) | Support agent |
| [`migration`](https://github.com/llmci-cli/llmci-testbed/tree/main/migration) | Model migration |

## Examples

| Example | What it demonstrates |
|---------|---------------------|
| [`01-ci-regression`](examples/01-ci-regression/) | Ticket classifier with exact_match + F1 |
| [`02-model-migration`](examples/02-model-migration/) | Prompt optimization across models |
| [`03-llm-as-judge`](examples/03-llm-as-judge/) | Open-ended generation with rubric judging |
| [`04-custom-judge`](examples/04-custom-judge/) | JSON schema validation with a Python judge |
| [`05-agent-single-turn`](examples/05-agent-single-turn/) | Tool-using agent with constraint checking |
| [`06-agent-multi-turn`](examples/06-agent-multi-turn/) | Multi-turn conversation testing |
| [`07-pipeline-level`](examples/07-pipeline-level/) | Full RAG pipeline end-to-end |
| [`08-fastapi-service`](examples/08-fastapi-service/) | Pre/post processing pipeline with dual-level testing |
| [`09-summarization-qa`](examples/09-summarization-qa/) | Multi-criteria LLM judge with reference-free evaluation |
| [`10-agent-openai-agents`](examples/10-agent-openai-agents/) | TraceBuilder + OpenAI Agents SDK adapter |

## CLI Reference

```
llmci run              Run evals and report results
llmci migrate          Optimize prompts for a new model
llmci init             Generate llmci.yaml interactively
llmci dataset init     Create a new eval dataset
llmci dataset add      Add examples interactively
llmci dataset check    Analyze dataset coverage
llmci dataset import   Import from CSV/JSON
llmci import-promptfoo Convert a Promptfoo config
```

Global flags: `-v` (verbose), `--debug` (full logging), `--version`.

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

Apache 2.0
