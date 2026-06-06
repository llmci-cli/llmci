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

For monorepos, discover configs and run them all:

```bash
llmci discover
llmci run --all
llmci run --all --root services/ticket-classifier
llmci run --all --include "services/**" --exclude "services/summarizer/llmci.yaml"
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
| `rag` | RAG pipelines (faithfulness, relevance, retrieval) | `judge: {type: rag, criteria: [...]}` |

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

**Cost / tokens (lower is better):**
- `cost_total`, `cost_mean` — total and per-example cost (USD), from litellm pricing
- `tokens_in_mean`, `tokens_out_mean`, `tokens_total_mean` — average token usage

For **direct** targets, cost and token usage are read from the provider response.
For **command** targets, your script can opt in by adding `"usage"` and `"cost"` to its
output JSON:

```json
{"output": "...", "usage": {"tokens_in": 1200, "tokens_out": 300}, "cost": 0.05}
```

Each metric supports two threshold modes:
- `absolute` — score must be >= threshold. For **lower-is-better** metrics (latency,
  cost, tokens, `error_rate`) the check inverts: value must be <= threshold.
- `max_regression` — regression from baseline must be <= threshold (e.g., 0.05 = max
  5%). A regression is a *drop* for higher-is-better metrics and a *rise* for
  lower-is-better metrics, so a cost increase past the threshold fails the gate.

## Output Formats

By default `llmci run` prints a markdown report (and posts it as a PR comment in
GitHub Actions). For other CI systems, emit a machine-readable format with
`--output-format`:

```bash
llmci run --output-format junit --output results.xml   # GitLab, Bitbucket, Azure DevOps, Jenkins, CircleCI
llmci run --output-format sarif --output results.sarif # code-scanning / inline annotations
llmci run --output-format json  --output results.json  # programmatic consumers
```

- **junit** — each eval is a `<testsuite>`, each metric a `<testcase>`; failed
  thresholds emit `<failure>`, and `max_regression` checks with no baseline emit
  `<skipped>`. Wire `results.xml` into your CI's native test reporting.
- **sarif** — SARIF 2.1.0; only failing thresholds become results (an empty list
  means clean), so it drops straight into code-scanning surfaces.
- **json** — structured per-eval metrics and threshold outcomes.

The PR comment always stays markdown regardless of `--output-format`.

## Response Caching

Re-running CI shouldn't re-pay for unchanged examples. For **direct API targets**,
llmci caches each response keyed on `(provider, model, prompt, input)` under
`.llmci/cache/responses/`:

```bash
llmci run                  # uses the cache; identical calls are free on re-run
llmci run --no-cache       # bypass the cache entirely
llmci run --refresh-cache  # ignore cached responses but refresh them with live calls
```

Command-mode targets are never cached (they may have side effects). Add
`.llmci/cache/` to `.gitignore`.

## Flake Resistance

LLM outputs are nondeterministic, so a single run can pass or fail a threshold by
chance. Run each eval over several rounds and gate on statistical significance so a
flaky result doesn't block (or sneak through) a PR:

```yaml
settings:
  samples_per_example: 5   # run each eval 5 rounds
  significance: 0.95       # confidence level for regression gating
```

Or from the CLI:

```bash
llmci run --samples 5 --significance 0.95 --compare-to=origin/main
```

When `samples_per_example > 1`:

- Each metric is **averaged across rounds** and reported with a confidence interval,
  e.g. `accuracy 0.562 [0.440, 0.685]`.
- For `max_regression` thresholds with `significance` set, a drop only **fails the
  gate when it exceeds the threshold beyond run-to-run noise** (the optimistic end of
  the confidence interval still breaches the threshold). Drops within noise are
  reported under "Regressions Within Noise (not enforced)" instead of failing.
- Sampling rounds bypass the response cache so each round is an independent draw.

## CI Integration

### GitHub Actions

Add to your workflow:

```yaml
- uses: llmci-cli/llmci@main
  with:
    compare-to: origin/main
    llmci-version: 0.1.9
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
- uses: llmci-cli/llmci@main
  with:
    config: services/api/llmci.yaml
    compare-to: origin/main
    llmci-version: 0.1.9
```

Or run every discovered config:

```yaml
- uses: llmci-cli/llmci@main
  with:
    all: "true"
    include: "services/**"
    exclude: "services/experimental/**"
    compare-to: origin/main
    llmci-version: 0.1.9
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

## RAG Evaluation

Score retrieval-augmented pipelines on RAG-specific dimensions. Each criterion
produces a 0–1 sub-score that is surfaced as a **gateable metric by name**, so you can
set independent thresholds on faithfulness, relevance, and retrieval quality:

```yaml
evals:
  - name: rag-qa
    dataset: ./evals/qa.jsonl
    judge:
      type: rag
      model: gpt-4o-mini        # used by the LLM-based criteria
      criteria:
        - {name: faithfulness,        type: faithfulness,        weight: 2.0}
        - {name: answer_relevance,    type: answer_relevance,    weight: 1.0}
        - {name: context_relevance,   type: context_relevance,   weight: 1.0}
        - {name: retrieval_recall,    type: retrieval_recall,    k: 5}
        - {name: retrieval_precision, type: retrieval_precision, k: 5}
    metrics:
      - {name: faithfulness,      threshold: 0.90, mode: absolute}
      - {name: retrieval_recall,  threshold: 0.80, mode: absolute}
      - {name: mean_score,        threshold: 0.05, mode: max_regression}
```

| Criterion | What it measures | Needs |
|-----------|------------------|-------|
| `faithfulness` | Is the answer grounded in the retrieved context? | LLM + `contexts` |
| `answer_relevance` | Does the answer address the question? | LLM |
| `context_relevance` | Is the retrieved context relevant to the question? | LLM + `contexts` |
| `retrieval_recall` | Fraction of gold documents retrieved (`@k`) | `retrieved_ids` + `relevant_ids` |
| `retrieval_precision` | Fraction of retrieved documents that are relevant (`@k`) | `retrieved_ids` + `relevant_ids` |

Your RAG pipeline runs as a **command target** and writes structured output JSON:

```json
{"output": "<answer>", "contexts": ["passage 1", "passage 2"], "retrieved_ids": ["doc3", "doc7"]}
```

Gold retrieval labels live on each dataset row as `relevant_ids`:

```json
{"input": "What is the capital of France?", "relevant_ids": ["doc1", "doc2"]}
```

The retrieval criteria (`retrieval_recall` / `retrieval_precision`) are deterministic
and need no API key; the faithfulness/relevance criteria call the configured judge model.

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

Key `run` flags: `--config`, `--all`, `--compare-to`, `--update-baseline`,
`--output`, `--output-format` (markdown/junit/sarif/json), `--no-cache`,
`--refresh-cache`, `--samples`, `--significance`, `--smoke`.

Global flags: `-v` (verbose), `--debug` (full logging), `--version`.

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

Apache 2.0
