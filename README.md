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
| `pairwise` | "Is the new output better than baseline?" (win rate) | `judge: {type: pairwise, model: gpt-4o}` |
| `safety` | PII leakage, toxicity, jailbreak resistance | `judge: {type: safety, criteria: [...]}` |
| `structured` | JSON output validates against a JSON Schema | `judge: {type: structured, json_schema: {...}}` |

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
llmci run --output-format html  --output report.html   # shareable report (upload as a CI artifact)
```

- **junit** — each eval is a `<testsuite>`, each metric a `<testcase>`; failed
  thresholds emit `<failure>`, and `max_regression` checks with no baseline emit
  `<skipped>`. Wire `results.xml` into your CI's native test reporting.
- **sarif** — SARIF 2.1.0; only failing thresholds become results (an empty list
  means clean), so it drops straight into code-scanning surfaces.
- **json** — structured per-eval metrics and threshold outcomes.
- **html** — a self-contained (inline-CSS) report with the summary table, regressions,
  and per-example results. No external assets, so it uploads cleanly as a CI artifact.

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

Baselines also store per-example outputs, so when a PR regresses, the report shows an
**Output Diffs vs Baseline** section — the baseline output next to the current output
for each regressed example (matched by input), in both the markdown and HTML reports.

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

## Pairwise / Preference Evaluation

For open-ended generation, "is this answer good?" is hard to score absolutely.
Pairwise judging asks the easier question — "is the new output **better** than the
previous one?" — and reports a **win rate** vs the baseline:

```yaml
evals:
  - name: support-replies
    dataset: ./evals/tickets.jsonl
    judge:
      type: pairwise
      model: gpt-4o
      rubric: "Which reply is more accurate, helpful, and on-policy?"   # optional criterion
    metrics:
      - {name: win_rate, threshold: 0.50, mode: absolute}   # new must win >= 50% of the time
```

```bash
llmci run --compare-to=origin/main
```

The judge compares each current output against the **baseline output** for the same
input (stored in the baseline — run `--update-baseline` on main first), scoring each
example 1.0 (win) / 0.5 (tie) / 0.0 (loss). The mean is exposed as the `win_rate`
metric. Newly added examples with no baseline output score a neutral 0.5. Combine with
`samples_per_example` for a confidence interval on the win rate.

**Position bias:** LLM judges tend to favor whichever answer appears first. By default the
pairwise judge runs each comparison in **both orders** and averages them, so a judge that
blindly prefers one position scores a neutral 0.5 instead of a false win. This doubles the
judge calls; set `position_swap: false` on the judge to disable it (e.g. when judge cost
dominates and you've verified low bias).

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

## Safety / Red-Team Assertions

Catch safety regressions — leaked PII, toxic output, or a prompt that starts complying
with disallowed requests — before they merge. The `safety` judge scores each criterion
where **higher is safer** and exposes it as a gateable metric by name:

```yaml
evals:
  - name: safety
    dataset: ./evals/redteam.jsonl
    judge:
      type: safety
      model: gpt-4o-mini          # used by the LLM-based criteria
      criteria:
        - {name: pii_leakage,          type: pii_leakage}            # deterministic
        - {name: toxicity,             type: toxicity}               # LLM
        - {name: jailbreak_resistance, type: jailbreak_resistance}   # LLM
    metrics:
      - {name: pii_leakage,          threshold: 1.00, mode: absolute}   # zero tolerance
      - {name: jailbreak_resistance, threshold: 0.95, mode: absolute}
      - {name: toxicity,             threshold: 0.02, mode: max_regression}
```

| Criterion | What it measures | Needs |
|-----------|------------------|-------|
| `pii_leakage` | Output is free of PII/secrets (1.0 clean, 0.0 leak) | nothing (deterministic) |
| `toxicity` | Output is free of toxic/unsafe content | LLM |
| `jailbreak_resistance` | Model refused/avoided a disallowed request | LLM |

`pii_leakage` is deterministic and needs no API key: it scans for emails, phone numbers,
SSNs, credit-card numbers, IPv4 addresses, and AWS keys. Narrow the scan with
`categories: [email, ssn]` on the criterion. The `toxicity` and `jailbreak_resistance`
criteria call the configured judge model — point your dataset's `input` at adversarial /
red-team prompts so the judge scores how the model handled them.

### Generating red-team prompts

Don't hand-write the adversarial inputs — generate them. `llmci redteam generate` expands
a few plain seed intents into many adversarially-framed prompts (jailbreak,
prompt-injection, PII-extraction, and obfuscation techniques), fully deterministically and
with no API key:

```bash
# See the built-in attack library
llmci redteam generate --list

# Expand seeds.txt into an adversarial dataset
llmci redteam generate \
  --seeds seeds.txt \
  --category pii_extraction --category injection \
  --output evals/attacks.jsonl
```

`seeds.txt` is one intent per line (or a `.jsonl` with an `input`/`seed`/`prompt` field).
Each generated row carries `attack`, `category`, and `seed` metadata so a failing gate can
attribute the leak to a specific technique. Filter with repeatable `--category` / `--attack`
flags, and add `--include-control` to also emit the raw seed as a baseline. Feed the output
straight into a `safety` judge (above) to gate it. See
[`examples/15-redteam`](examples/15-redteam/) for the full generate-then-gate flow.

## Structured-Output Evaluation

When a feature must emit machine-readable JSON (tool calls, extraction, config
generation), gate on validity with the built-in `structured` judge — no API key, fully
deterministic. It parses the output and validates it against a JSON Schema, scoring 1.0
when valid and 0.0 otherwise:

```yaml
judge:
  type: structured
  json_schema:                 # inline, or a path: json_schema: ./schema.json
    type: object
    required: [id, name, price]
    additionalProperties: false
    properties:
      id:    {type: integer}
      name:  {type: string, minLength: 1}
      price: {type: number, minimum: 0}
metrics:
  - {name: accuracy, threshold: 1.0, mode: absolute}
```

The self-contained validator supports the practical JSON-Schema subset: `type` (incl.
lists of types), `required`, `properties`, `additionalProperties`, `items`, `enum`,
`minimum`/`maximum`, `minLength`/`maxLength`, `minItems`/`maxItems`, and `pattern`. Set
`partial_credit: true` to score the **fraction** of required top-level fields that validate
instead of pass/fail. See [`examples/16-structured-output`](examples/16-structured-output/).

## Judge Calibration & Drift

An LLM judge is only worth gating on if it agrees with humans — and judges drift
silently when you bump the judge model. `llmci judge calibrate` measures both:

```bash
llmci judge calibrate --eval support-replies --labels labels.jsonl --save-snapshot
```

The labeled set is JSONL where each row carries the output to score and a human label
(`1`/`0`, `true`/`false`, `pass`/`fail`, or a float in `[0, 1]`):

```json
{"input": "How do I reset my password?", "output": "Click 'Forgot password'…", "human_score": 1}
{"input": "Is my data encrypted?", "output": "idk", "human_score": 0}
```

It runs the eval's configured judge over those examples and reports agreement:

| Metric | Meaning |
|--------|---------|
| Agreement rate | Fraction where judge and human agree on pass/fail (threshold 0.5) |
| Cohen's kappa | Agreement beyond chance (`slight` … `almost perfect`) |
| Mean abs error | Average distance between judge and human scores |
| Pearson r | Correlation between judge and human scores |

`--save-snapshot` records the judge model and its per-example scores under
`.llmci/calibration/<eval>.json`. A later run compares against that snapshot and reports
**drift** — the mean change in scores on the same labeled set — flagging when the judge
model changed. Gate it in CI:

```bash
llmci judge calibrate --eval support-replies --labels labels.jsonl \
  --min-agreement 0.80 --max-drift 0.10
```

`--min-agreement` fails when judge↔human agreement drops too low; `--max-drift` fails
when a judge-model change shifts scores more than allowed.

## Extending llmci: Judge, Metric & Report Plugins

Need domain-specific scoring? Register a new `judge.type` without forking. A plugin is a
`Judge` subclass (or a `(JudgeConfig) -> Judge` factory) registered with
`register_judge`:

```python
# my_repo/eval_plugins.py
from llmci.judges.base import Judge
from llmci.models import JudgeResult
from llmci.plugins import register_judge


class SqlValidityJudge(Judge):
    async def evaluate_single(self, input: str, expected: str, actual: str) -> JudgeResult:
        ok = is_valid_sql(actual)
        return JudgeResult(score=1.0 if ok else 0.0, reason=None if ok else "invalid SQL")


register_judge("sql_validity", SqlValidityJudge)
```

**Local plugins** — list the module under `plugins:` so it's imported at config load:

```yaml
plugins:
  - my_repo.eval_plugins

evals:
  - name: text2sql
    dataset: ./evals/queries.jsonl
    judge: {type: sql_validity}
```

**Distributable plugins** — ship a package that advertises the judge via the
`llmci.judges` entry-point group; it's discovered automatically once installed:

```toml
# pyproject.toml of your plugin package
[project.entry-points."llmci.judges"]
sql_validity = "my_pkg.judges:SqlValidityJudge"
```

Plugin types are validated when the judge is built and may not shadow a built-in type.

### Custom metrics

Register a custom **metric** the same way. A metric function takes a `MetricContext`
(examples, target results, judge results, and the indices/scores of non-errored examples)
and returns one aggregate float — then it's gateable by name like any built-in:

```python
from llmci.plugins import MetricContext, register_metric


def answer_length(ctx: MetricContext) -> float:
    lengths = [len(ctx.results[i].output) for i in ctx.valid_indices]
    return sum(lengths) / len(lengths) if lengths else 0.0


register_metric("answer_length", answer_length, lower_is_better=True)
```

```yaml
metrics:
  - {name: answer_length, threshold: 600, mode: absolute}   # avg chars must stay <= 600
```

Pass `lower_is_better=True` to flip the threshold direction (like cost/latency). Metric
plugins also load from the `llmci.metrics` entry-point group for distribution. See
[`examples/13-plugin-judge`](examples/13-plugin-judge/), which registers both a judge and
a metric.

### Custom report sinks

Register a **report sink** to ship results somewhere after each run — a Slack message, a
dashboard, an artifact upload. A sink receives a `ReportContext` (the eval results, the
configs, the overall `passed` flag, and the rendered markdown) and runs for its side
effect. List it under `reporters:` to activate it:

```python
from llmci.plugins import ReportContext, register_reporter


def slack_sink(ctx: ReportContext) -> None:
    status = "passed" if ctx.passed else "FAILED"
    post_to_slack(f"llmci {status} ({len(ctx.results)} evals)\n{ctx.report_markdown}")


register_reporter("slack", slack_sink)
```

```yaml
plugins: [my_sinks]      # module that calls register_reporter
reporters: [slack]       # sinks to invoke after the run
```

Sinks load from local modules (via `plugins:`) or the `llmci.reporters` entry-point group.
A sink that raises only warns — it never changes the pass/fail gate.

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
| [`11-safety-pii`](examples/11-safety-pii/) | Safety judge with a deterministic PII-leakage gate |
| [`12-rag-retrieval`](examples/12-rag-retrieval/) | RAG judge with deterministic retrieval recall/precision |
| [`13-plugin-judge`](examples/13-plugin-judge/) | Custom judge type registered via the plugin API |
| [`14-judge-calibration`](examples/14-judge-calibration/) | `judge calibrate`: judge↔human agreement + drift |
| [`15-redteam`](examples/15-redteam/) | `redteam generate`: adversarial dataset gated by the safety judge |
| [`16-structured-output`](examples/16-structured-output/) | `structured` judge: validate JSON output against a JSON Schema |

Examples 11–16 are fully deterministic and run with **no API key** — handy for trying
the safety, RAG, plugin, calibration, red-team, and structured-output features locally.

## CLI Reference

```
llmci run              Run evals and report results
llmci migrate          Optimize prompts for a new model
llmci judge calibrate  Measure judge↔human agreement and detect drift
llmci redteam generate Generate an adversarial dataset to probe safety
llmci init             Generate llmci.yaml interactively
llmci dataset init     Create a new eval dataset
llmci dataset add      Add examples interactively
llmci dataset check    Analyze dataset coverage
llmci dataset import   Import from CSV/JSON
llmci import-promptfoo Convert a Promptfoo config
```

Key `run` flags: `--config`, `--all`, `--compare-to`, `--update-baseline`,
`--output`, `--output-format` (markdown/junit/sarif/json/html), `--no-cache`,
`--refresh-cache`, `--samples`, `--significance`, `--smoke`.

Global flags: `-v` (verbose), `--debug` (full logging), `--version`.

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

Apache 2.0
