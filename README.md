# Scaffold

CI-native regression testing and migration for LLMs.

Catch quality drops before they merge. Migrate models without breaking things.

Scaffold is not an observability tool — it's a **pre-merge safety gate**. Define eval datasets, set quality thresholds, and let CI block bad changes to your prompts, models, or pipelines.

## Installation

```bash
pip install scaffold-ai
```

Requires Python 3.11+.

## Quick Start

### 1. Initialize

```bash
scaffold init
```

This creates a `scaffold.yaml` config and a starter eval dataset. You'll be asked:
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
scaffold dataset add --name my-eval
```

### 3. Run

```bash
scaffold run
```

Output:

```
## Scaffold Eval Report

| Eval | Metric | Score | Threshold | Status |
|------|--------|-------|-----------|--------|
| ticket-classification | accuracy | 0.950 | ≥ 0.9 | ✅ |
| ticket-classification | f1_macro | 0.940 | ≥ 0.85 | ✅ |
```

Exit code 0 = all thresholds pass. Exit code 1 = regression detected.

## Configuration

`scaffold.yaml` defines your target, evals, and settings:

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
- uses: alexminnaar/scaffold@main
  with:
    compare-to: origin/main
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Or use the CLI directly:

```yaml
- run: pip install scaffold-ai
- run: scaffold run --compare-to=origin/main
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

When running in GitHub Actions, Scaffold automatically posts eval results as a PR comment.

### Baselines

Store baseline scores on your main branch:

```bash
scaffold run --update-baseline
```

Then compare PRs against that baseline:

```bash
scaffold run --compare-to=main
```

## Model Migration

When switching models (e.g., GPT-4o to GPT-4.5), Scaffold can automatically tune your prompt to maintain quality parity:

```bash
scaffold migrate \
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

Supports:
- **Single-turn** and **multi-turn** conversations
- **Constraint checking** — tool call budgets, required/forbidden tools, token limits
- **Outcome judging** — LLM-based evaluation of final output
- **Trajectory judging** — LLM-based evaluation of execution path quality
- **Full replay** or **history injection** modes for multi-turn

## Dataset Tools

```bash
# Initialize a new dataset
scaffold dataset init --name my-eval --type classification

# Add examples interactively
scaffold dataset add --name my-eval

# Analyze coverage and quality
scaffold dataset check --name my-eval

# Import from CSV or JSON
scaffold dataset import --name my-eval --from data.csv
```

## Migrating from Promptfoo

```bash
scaffold import-promptfoo promptfooconfig.yaml
```

Converts providers, test assertions, and variables into Scaffold's format.

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

## CLI Reference

```
scaffold run              Run evals and report results
scaffold migrate          Optimize prompts for a new model
scaffold init             Generate scaffold.yaml interactively
scaffold dataset init     Create a new eval dataset
scaffold dataset add      Add examples interactively
scaffold dataset check    Analyze dataset coverage
scaffold dataset import   Import from CSV/JSON
scaffold import-promptfoo Convert a Promptfoo config
```

Global flags: `-v` (verbose), `--debug` (full logging), `--version`.

## License

Apache 2.0
