# Scaffold: CI-Native Regression Testing & Migration for LLMs

**One-liner:** Catch LLM quality drops before they merge. Migrate models without breaking things.

---

## Problem

Teams using LLMs in production face two recurring pain points:

1. **Content test fragility.** When anyone changes a prompt, model, model parameters, or upstream pipeline code, there's no standardized way to verify that the change didn't degrade output quality. Teams either build custom eval harnesses (inconsistently, per-team) or skip testing entirely.

2. **Model migration friction.** When a model is upgraded or retired, teams must manually re-tune prompts to achieve parity on the new model. This is tedious, error-prone, and often blocks migrations for weeks.

## Solution

A CLI tool that engineers plug into CI/CD with minimal effort. It runs a customer-provided eval dataset against their LLM pipeline, detects quality regressions, and gates PRs accordingly. It also automates prompt re-tuning during model migrations.

**This is not observability.** It's a safety gate — closer to a test suite than a dashboard.

---

## Core Concepts

### Two Categories of LLM Tasks

| Category | Examples | Gold labels | Evaluation method |
|---|---|---|---|
| **Deterministic-output** | Classification, extraction, structured output, routing | Objectively correct answers | Standard metrics (F1, accuracy, precision, recall, exact match) |
| **Open-ended generation** | Summarization, rewriting, conversation, creative content | Reference outputs (not uniquely correct) | LLM-as-judge with rubric-based criteria |

### Two Testing Levels

| Level | Input represents | Catches | Trade-offs |
|---|---|---|---|
| **Prompt-level** (unit test) | Fully-constructed prompt text | Prompt changes, model changes, param changes | Fast, hermetic, no external deps. Misses upstream changes. |
| **Pipeline-level** (integration test) | Raw user input / system entry point | Everything that affects output (including upstream data/retrieval/preprocessing changes) | Slower, may depend on services. More comprehensive. |

Both levels should be supported. Pipeline-level is the default because it catches the sneaky case where the prompt is unchanged but upstream code altered what gets fed into it.

### Two Threshold Modes

- **Absolute:** "F1 must be above 0.93, period." Clear quality floor.
- **Relative (max regression):** "Score must not drop more than X% from the main branch baseline." Answers the real question: did *this PR* make things worse?

### Trigger Strategy

Default: run on every PR. The whole value proposition is catching regressions before merge. Customers who want to optimize cost can:

- Use `smoke_test_size` to run a small subset on every PR, full suite on merge to main or nightly.
- Configure `trigger.paths` to only run when specific file paths are touched (opt-in, not default).

---

## Config Format

The primary interface. Lives at `llmci.yaml` in the repo root.

```yaml
version: 1

target:
  # Black-box mode: scaffold invokes customer's pipeline as a subprocess
  command: "python run_prompt.py --input {input_file} --output {output_file}"
  # OR direct API mode:
  # provider: openai
  # model: gpt-4o
  # prompt_file: ./prompts/classify.txt

evals:
  - name: ticket-classification
    level: prompt            # or "pipeline"
    dataset: ./evals/ticket_classification.jsonl
    judge: exact_match
    metrics:
      - name: f1_macro
        threshold: 0.93
        mode: absolute
      - name: accuracy
        threshold: 0.02
        mode: max_regression

  - name: response-quality
    level: pipeline
    dataset: ./evals/response_quality.jsonl
    judge:
      type: llm
      model: gpt-4o
      rubric:
        - id: factual_accuracy
          prompt: "Does the response contain only factually correct information?"
        - id: completeness
          prompt: "Does the response address all parts of the user's question?"
    metrics:
      - name: rubric_pass_rate
        threshold: 0.03
        mode: max_regression

  - name: json-schema-compliance
    level: pipeline
    dataset: ./evals/structured_output.jsonl
    judge:
      type: custom
      module: ./judges/schema_judge.py
      function: evaluate
    metrics:
      - name: pass_rate
        threshold: 0.95
        mode: absolute

settings:
  parallelism: 10
  timeout_per_call: 30
  retries: 2
  smoke_test_size: 50
```

### Built-in and Custom Judges

llmci ships with judges for common evaluation patterns, but teams can also define their own:

| Judge type | How it works | When to use |
|---|---|---|
| **`exact_match`** | Output must exactly match the expected value | Classification, extraction, routing |
| **`metrics`** (F1, accuracy, precision, recall) | Standard ML metrics computed over the full dataset | Multi-class classification, NER |
| **`llm`** (LLM-as-judge) | An LLM evaluates outputs against a rubric | Open-ended generation, summarization, conversation |
| **`custom`** | User-defined Python function | Domain-specific logic: JSON schema validation, business rule checks, regex patterns, multi-field comparison |

Custom judges are Python functions that receive the input, expected output, and actual output, and return a score:

```python
# judges/schema_judge.py
import json

def evaluate(input: str, expected: str, actual: str) -> dict:
    """Check that the output is valid JSON matching the expected schema."""
    try:
        parsed = json.loads(actual)
        has_required_fields = all(k in parsed for k in ["category", "confidence", "reasoning"])
        confidence_valid = 0.0 <= parsed.get("confidence", -1) <= 1.0
        return {
            "score": 1.0 if (has_required_fields and confidence_valid) else 0.0,
            "reason": "Valid" if has_required_fields else f"Missing fields: {[k for k in ['category', 'confidence', 'reasoning'] if k not in parsed]}"
        }
    except json.JSONDecodeError as e:
        return {"score": 0.0, "reason": f"Invalid JSON: {e}"}
```

This is referenced from the config as `judge: {type: custom, module: ./judges/schema_judge.py, function: evaluate}`. Custom judges run locally alongside Scaffold — no LLM calls, no latency, no cost for deterministic checks.

### LLM Provider Authentication

**llmci never stores, manages, or proxies API keys.** It reads standard environment variables set by the user. This is a deliberate design choice — zero auth infrastructure, zero credential storage, zero security surface.

There are up to three places where Scaffold needs LLM access:

| Role | What it's for | Who picks the model |
|---|---|---|
| **Target model** | The LLM being evaluated (in direct API mode) | User sets in `llmci.yaml` under `target.model` |
| **Judge model** | LLM-as-judge for rubric evaluation | User sets in `llmci.yaml` under `evals[].judge.model` (per eval) |
| **Optimizer model** | The LLM that rewrites prompts during migration | User sets via `llmci migrate --optimizer-model` |

In **command mode** (black-box), llmci doesn't call any LLM for the target at all — the user's command handles that internally. Scaffold only needs LLM access for the judge and optimizer, if used.

**How auth works in practice:**

```bash
# Locally — user has keys in their environment
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
llmci run
```

```yaml
# In CI — keys come from the CI platform's secret management
# GitHub Actions
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

# GitLab CI — same pattern with CI/CD variables
# Jenkins — same pattern with credentials plugin
```

llmci uses [litellm](https://github.com/BerriAI/litellm) under the hood for direct API calls, which means it inherits litellm's provider support and env var conventions automatically. Any provider litellm supports, llmci supports — OpenAI, Anthropic, Google, Mistral, Azure, AWS Bedrock, local models via Ollama, or any OpenAI-compatible endpoint.

The config specifies model names using litellm's naming convention:

```yaml
target:
  provider: anthropic
  model: claude-sonnet-4-20250514

evals:
  - name: quality-check
    judge:
      type: llm
      model: openai/gpt-4o      # litellm prefix for explicit provider routing

# Or target a local model via Ollama
# target:
#   provider: ollama
#   model: llama3
```

**This means users can mix providers freely** — evaluate a Claude model using GPT-4o as the judge, or migrate from an OpenAI model to a Mistral model with Claude as the optimizer. No lock-in at any layer.

### Eval Dataset Schema

JSONL with two required fields. Additional fields are passed through to the customer's command.

```jsonl
{"input": "My printer won't connect to wifi", "expected": "hardware"}
{"input": "I need a refund for order #882", "expected": "billing"}
```

For LLM-as-judge evals, `expected` is a reference response:

```jsonl
{"input": "Summarize this article: ...", "expected": "The article discusses..."}
```

### Eval Dataset Storage

**Default: in-repo.** The eval dataset is a test fixture like any other — version-controlled, auditable, travels with the code. For most teams (a few hundred to a few thousand JSONL rows, single-digit MB), this works perfectly.

**When in-repo breaks down:**

| Concern | Situation | Solution |
|---|---|---|
| **Size** | Large datasets (50MB+) with rich context (full documents, conversation histories) degrade git performance | Remote source with local caching |
| **Sensitive data** | Eval data derived from production contains PII or proprietary content that shouldn't be committed, even to a private repo | Remote source behind access controls |
| **Shared datasets** | Multiple repos/teams eval against the same gold dataset; duplication leads to drift | Single remote source, referenced from all repos |

**Config supports both local and remote sources:**

```yaml
evals:
  - name: ticket-classification
    # Local path (default, checked into repo)
    dataset: ./evals/ticket_classification.jsonl

  - name: response-quality
    # Remote source (pulled at eval time)
    dataset:
      source: s3://company-evals/response_quality_v3.jsonl
      # also supported: gs://, https://
      cache: true  # cache locally to avoid re-downloading every run
```

**v1:** Support local paths only. Design the config schema to accommodate remote sources without a breaking change.

**Later:** Add remote source support (S3, GCS, HTTPS). This also becomes a natural feature of the paid hosted tier — managed dataset storage with versioning and access control.

---

## Architecture

### PR Flow

```
Developer pushes branch
        │
        ▼
CI triggers (GitHub Actions / GitLab CI / etc.)
        │
        ▼
llmci run --compare-to=main
        │
        ├── 1. Load eval config (llmci.yaml)
        ├── 2. Load eval datasets (.jsonl files)
        ├── 3. Execute LLM calls (branch version)
        ├── 4. Compute metrics
        ├── 5. Load baseline scores (from main branch)
        ├── 6. Compare: regression detected?
        │
        ▼
Post result as PR check + comment
        │
    ┌───┴───┐
    │       │
  PASS    FAIL
  (merge)  (block + show diff report)
```

### Baseline Storage (v1: In-Repo)

```
.llmci/
  baselines/
    ticket-classification.json    # {"f1_macro": 0.95, "accuracy": 0.97}
    response-quality.json         # {"rubric_pass_rate": 0.91}
```

- Committed to the repo, version-controlled, auditable.
- Updated on main branch via `llmci run --update-baseline`.
- PRs compare their results against baselines from the base branch using `git show main:.llmci/baselines/...`.
- Fully stateless — no external service needed.

### PR Report

Posted as a PR comment and/or CI summary:

```
## llmci Eval Report

| Eval                  | Metric           | Baseline | This PR | Threshold | Status |
|-----------------------|------------------|----------|---------|-----------|--------|
| ticket-classification | f1_macro         | 0.951    | 0.948   | ≥ 0.93    | ✅      |
| ticket-classification | accuracy         | 0.972    | 0.943   | ≤ 2% drop | ❌      |
| response-quality      | rubric_pass_rate | 0.912    | 0.901   | ≤ 3% drop | ✅      |

### Regressions Detected

**ticket-classification / accuracy**: dropped 2.9% (0.972 → 0.943, threshold: 2%)

<details>
<summary>Degraded examples (7 of 243)</summary>

| Input (truncated)              | Expected  | Got       |
|--------------------------------|-----------|-----------|
| "My printer won't connect..."  | hardware  | software  |
| "Refund for order #882..."     | billing   | general   |
</details>
```

---

## Model Migration

### The Optimization Loop

Migration is modeled as a prompt optimization problem, analogous to gradient descent:

| ML Training Concept | Prompt Migration Equivalent |
|---|---|
| Learning rate / step size | How many changes the optimizer LLM makes per iteration |
| Training set | Eval examples used to score each iteration |
| Validation set | Separate examples checked each iteration for early stopping + overfitting detection |
| Holdout set | Examples only evaluated at the end — the honest final score |
| Loss function | The eval metric (F1, rubric pass rate, etc.) |
| Early stopping | Halt when improvement plateaus for N iterations |
| Overfitting | Prompt gets tuned to pass specific examples but fails on unseen inputs |

### Data Splitting

```
Customer's eval dataset (100%)
        │
        ├── 70%  Train set       → used by optimizer to score iterations
        ├── 15%  Validation set   → checked each iteration, used for early stopping
        └── 15%  Holdout set      → evaluated only at the end, reported as final score
```

Default: automatic random split with fixed seed for reproducibility. Customer can provide explicit splits if desired.

### Step Size Control

The optimizer LLM is explicitly constrained to make minimal changes per iteration:

- System prompt instructs: "change as little as possible, do not rewrite from scratch, prefer rewording existing instructions over adding new ones."
- Optionally enforce mechanically: reject proposed prompts where the edit distance exceeds a threshold.
- Small steps aid debuggability — if one change improves the score, you know exactly what worked.

### Early Stopping

```
patience = 3            # stop after N iterations with no improvement
min_improvement = 0.5%  # minimum improvement to count
max_iterations = 20     # hard cap

best_val_score = initial_score
stale_count = 0

for each iteration:
    new_prompt = optimizer.suggest_modification(current_prompt, failures)
    train_score = evaluate(new_prompt, train_set)
    val_score = evaluate(new_prompt, validation_set)

    if val_score > best_val_score + min_improvement:
        best_val_score = val_score
        best_prompt = new_prompt
        stale_count = 0
    else:
        stale_count += 1

    if stale_count >= patience:
        break

holdout_score = evaluate(best_prompt, holdout_set)
```

### Optimizer Model

Default: use a strong model (customer-configurable) as the optimizer, separate from the target model. The optimizer is a development-time tool, not a production dependency, so using a more capable/expensive model is acceptable.

### Migration Report

```
## Migration Report: gpt-4o → gpt-4.5

### Optimization Summary
- Iterations: 8 (stopped early, patience=3)
- Train score:      0.891 → 0.947
- Validation score: 0.883 → 0.941
- Holdout score:    0.879 → 0.938 (original baseline on old model: 0.952)

### Prompt Diff
- Line 12: "Classify the following into exactly one category"
+ Line 12: "Classify the following into exactly one category. Choose the
+           single most specific category that applies."

### Remaining Regressions (holdout set)
5 of 87 holdout examples still regressed. Top failure patterns:
- 3/5: ambiguous inputs where old model defaulted to "general"
- 2/5: edge cases with multilingual input
```

---

## CLI Surface

```bash
# Initialize a new scaffold config interactively
llmci init

# Run evals and compare against a baseline branch
llmci run [--compare-to=main] [--smoke] [--output=report.md]

# Update the stored baseline (run on main branch in CI)
llmci run --update-baseline

# Run prompt migration optimization
llmci migrate --from <model> --to <model> --eval <eval_name>

# Create and manage eval datasets
llmci dataset init --name <eval_name> --type <deterministic|open_ended|agent>
llmci dataset add --name <eval_name>                # interactive example entry
llmci dataset check --name <eval_name>              # coverage analysis + gap detection
llmci dataset import --name <eval_name> --from <file.csv|file.json>

# Generate / expand eval datasets (v2)
scaffold generate --from-logs <exported-logs-dir/> --output <dataset.jsonl>
scaffold generate --from-spec <agent_config.yaml> --output <dataset.jsonl>
scaffold generate --augment <seed.jsonl> --output <expanded.jsonl> --target-size 200

# Import config from Promptfoo
llmci import-promptfoo <promptfooconfig.yaml>
```

---

## CI Integration

### GitHub Actions

```yaml
# .github/workflows/scaffold.yml
name: LLM Content Tests
on: [pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install llmci
      - run: llmci run --compare-to=origin/main
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}  # if using multiple providers
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

The `llmci run` command exits 0 (pass) or 1 (fail), so it works as a CI gate without any special integration. The report action is optional sugar.

---

## Tech Stack & Package Structure

**Language:** Python. The target audience is Python-native, LLM SDKs are Python-first, and extensibility (custom judges as Python functions) is strongest here.

**Language-agnostic by design.** While Scaffold itself is Python, the `command` target mode makes it language-agnostic. Any language, any framework, any pipeline — as long as it reads an input file and writes an output file, llmci can evaluate it. This is a key differentiator against DeepEval (pytest-coupled, Python-only) and DSPy (requires rewriting as DSPy programs). A Node.js service, a Go binary, a bash script — they all work.

**Distribution:** PyPI (`pip install llmci`), GitHub Action wrapper, Docker image.

```
llmci/
├── pyproject.toml
├── src/
│   └── scaffold/
│       ├── cli.py                  # CLI entry point (click or typer)
│       ├── config.py               # llmci.yaml parsing + validation
│       ├── runner.py               # orchestrates eval execution
│       ├── targets/
│       │   ├── command.py          # subprocess-based (black box mode)
│       │   └── direct.py           # direct API calls via litellm
│       ├── judges/
│       │   ├── exact_match.py
│       │   ├── metrics.py          # f1, accuracy, precision, recall
│       │   ├── llm_judge.py        # rubric-based LLM-as-judge
│       │   └── custom.py           # user-defined judge functions
│       ├── baseline.py             # baseline storage + comparison
│       ├── dataset/
│       │   ├── init.py             # llmci dataset init
│       │   ├── add.py              # interactive example entry
│       │   ├── check.py            # coverage analysis + gap detection
│       │   └── import_data.py      # import from CSV/JSON
│       ├── generate/                   # v2: dataset generation
│       │   ├── from_logs.py            # production log import
│       │   ├── from_spec.py            # scenario generation from agent specs
│       │   └── augment.py              # perturbation-based expansion
│       ├── migrate/
│       │   ├── optimizer.py        # prompt optimization loop
│       │   ├── splitter.py         # train/validation/holdout splitting
│       │   └── stopping.py         # early stopping logic
│       ├── report.py               # markdown / CI report generation
│       └── integrations/
│           ├── github.py           # PR comment posting
│           ├── gitlab.py
│           ├── openai_agents.py    # v2: OpenAI Agent SDK adapter
│           ├── pydantic_ai.py      # v2: PydanticAI adapter
│           └── claude_agents.py    # v2: Claude Agent SDK adapter
├── action.yml                      # GitHub Action definition
├── examples/
│   ├── 01-ci-regression/           # Ticket classifier with exact_match + F1
│   │   ├── README.md
│   │   ├── llmci.yaml
│   │   ├── evals/tickets.jsonl
│   │   ├── prompts/classify.txt
│   │   └── run_prompt.py
│   ├── 02-model-migration/         # Migrate GPT-4o → GPT-4.5
│   │   ├── README.md
│   │   ├── llmci.yaml
│   │   ├── evals/tickets.jsonl
│   │   └── prompts/classify.txt
│   ├── 03-llm-as-judge/            # Open-ended generation with rubric judging
│   │   ├── README.md
│   │   ├── llmci.yaml
│   │   ├── evals/responses.jsonl
│   │   └── run_prompt.py
│   ├── 04-custom-judge/            # JSON schema validation with a Python judge
│   │   ├── README.md
│   │   ├── llmci.yaml
│   │   ├── evals/structured.jsonl
│   │   ├── judges/schema_judge.py
│   │   └── run_prompt.py
│   ├── 05-agent-single-turn/       # Single-turn agent with composite judging
│   │   ├── README.md
│   │   ├── llmci.yaml
│   │   ├── evals/scenarios.jsonl
│   │   └── run_agent.py
│   ├── 06-agent-multi-turn/        # Multi-turn conversation testing
│   │   ├── README.md
│   │   ├── llmci.yaml
│   │   ├── evals/conversations.jsonl
│   │   └── run_agent.py
│   └── 07-pipeline-level/          # Full pipeline test (RAG + LLM)
│       ├── README.md
│       ├── llmci.yaml
│       ├── evals/questions.jsonl
│       └── run_pipeline.py
└── tests/
```

Each example is self-contained and runnable. A user clones the repo, `cd`s into an example folder, sets their API key, and runs `llmci run`. The README in each example explains what it demonstrates, what to look for in the output, and how to adapt it to their own use case. The use-cases page (`use-cases.html`) links directly to these examples on GitHub.

---

## Build Sequence

| Phase | Scope | Timeline |
|---|---|---|
| **1. Core eval loop** | Config parsing, dataset loading, command + direct mode targets, deterministic judges (exact match, F1, accuracy), `llmci run`, `llmci dataset init/add/check` | Week 1–2 |
| **2. Baseline + CI** | In-repo baselines, `--compare-to` regression detection, `--update-baseline`, markdown report, GitHub Action, PR comments | Week 3 |
| **3. LLM-as-judge** | Rubric-based judging, configurable judge model, result caching | Week 4 |
| **4. Migration** | Train/val/holdout splitter, optimizer loop with step size control, early stopping, migration report, `llmci migrate` | Week 5–6 |
| **5. Polish + launch** | `llmci init`, documentation, runnable examples (one per use case), error handling, open source release, landing page screenshots (GitHub Actions check, PR comment with eval report, CI log output) | Week 7 |
| **5b. Promptfoo migration** | `llmci import-promptfoo` to convert Promptfoo YAML configs to llmci.yaml. Low effort, high adoption value — captures the migration wave. | Week 7 |

---

## Business Model

**Open core.**

- **Free (open source):** The full CLI — eval loop, CI integration, migration tool.
- **Paid (hosted service, later):** Remote baseline storage, dashboard with eval history & trends, team features (shared datasets, access control), managed judge API, migration-as-a-service.

---

## Competitive Landscape

*See `COMPETITIVE_ANALYSIS.md` for full profiles and feature gap analysis.*

### Market Map

| | Primary Use | CI Gate | Migration | Agent Eval | Dataset Gen |
|---|---|---|---|---|---|
| **Scaffold** | PR quality gate | Native | Automated | Full (trajectory + constraints) | Yes |
| **Promptfoo** *(now OpenAI)* | Prompt comparison + red teaming | Supported | No | No | No |
| **Braintrust** | Eval platform + observability | GitHub Action | No | Tracing only | No |
| **Langfuse** | Observability + tracing | Limited | No | Tracing only | No |
| **DeepEval** | Automated eval CI | Native | No | No | SaaS only |
| **DSPy** | Prompt optimization framework | No | Re-optimize | No | No |
| **Arize Phoenix** | Eval + production monitoring | No | No | Trace support | No |
| **RAGAS** | RAG evaluation | No | No | No | No |

### Key Competitive Dynamics

1. **Promptfoo acquisition by OpenAI (March 2026).** Promptfoo was the closest tool to llmci — CLI-native, YAML configs, CI support. The acquisition creates a trust gap for multi-provider teams, a migration wave of users seeking alternatives, and a strategic pivot risk as OpenAI may focus Promptfoo on red-teaming/security over general eval. Scaffold should explicitly position as provider-neutral and community-owned.

2. **Scaffold's unique moats.** No competitor offers automated model migration with holdout validation. No competitor does CI-gated agentic trajectory evaluation. No competitor combines dataset generation (from production logs, specs, and augmentation) with CI gating. These are not incremental features — they represent categories that don't exist in the competitive set.

3. **Complementary, not competitive, with observability.** Langfuse, Arize Phoenix, and Braintrust monitor production. Scaffold gates pre-merge. Teams should use both. This "composable tool" positioning (Unix philosophy) resonates with engineers who don't want to adopt a full platform for a CI check.

4. **DeepEval is the closest philosophical match.** It's CI-focused and open source. Key differentiators against DeepEval: pipeline-level testing (not just prompt-level), relative regression thresholds, migration automation, language-agnostic command mode (DeepEval is pytest-coupled), and agentic evaluation.

5. **Agentic evaluation is wide open.** Despite the explosion of agent frameworks (OpenAI Agent SDK, Claude Agent SDK, PydanticAI), nobody is doing CI-gated agent regression testing, trajectory quality evaluation, or agent-specific migration. This is Scaffold's biggest opportunity for differentiation.

---

## Go-to-Market

### Positioning

**"CI-native regression testing for LLMs. Provider-neutral. No platform required."**

llmci is a safety gate, not a dashboard. It catches regressions before merge, automates model migration, and works with every provider. It is not owned by any model provider.

### Target Personas

1. **ML/AI engineers** maintaining LLM-powered features who are tired of manual testing or no testing at all.
2. **Platform/infra engineers** who want to standardize LLM testing across teams (one config format, one CI step, every repo).
3. **Teams migrating off Promptfoo** due to the OpenAI acquisition — looking for a provider-neutral alternative with more features.

### Wedge Use Cases (easiest adoption paths)

1. **Model migration.** Team needs to upgrade a model and wants automated prompt re-tuning. Immediate, concrete, quantifiable value. This is Scaffold's strongest wedge because no alternative exists.
2. **Post-Promptfoo migration.** Team is evaluating alternatives after the acquisition. llmci offers a familiar CLI + YAML workflow with capabilities Promptfoo never had.
3. **First CI gate.** Team has no LLM testing at all. `llmci init` → `llmci run` in five minutes. The bar is zero — any testing is an improvement.

### Key Messages by Competitor

| When competing against | Lead with |
|---|---|
| **Promptfoo** | Provider neutrality, migration automation, pipeline-level testing, agentic eval |
| **Braintrust** | Zero infrastructure, no platform lock-in, open source, migration automation |
| **Langfuse** | Purpose-built for CI (not observability), migration automation, agentic eval |
| **DeepEval** | Pipeline-level testing, migration automation, language-agnostic command mode |
| **DSPy** | CI integration, holdout-validated migration, agentic trajectory eval, not a framework rewrite |
| **"We don't test"** | Five-minute setup, catches regressions before users do, model migration without the fire drill |

---

## v2: Agentic Systems

### Why Agents Are Harder

Single LLM calls have a clear input → output contract. Agentic systems (OpenAI Agent SDK, Claude Agent SDK, PydanticAI, etc.) introduce sequential decision-making — tool calls, routing, looping, branching — where the same input can take completely different execution paths.

This changes the eval problem in several ways:

| Dimension | Single LLM call | Agentic system |
|---|---|---|
| **What to evaluate** | Final output | Final outcome + trajectory quality + tool use + cost/latency + safety |
| **Non-determinism** | Moderate (stochastic output) | High (different paths each run, compounding variance) |
| **Gold dataset** | Input → expected output | Scenario → expected outcome + constraints |
| **Conversation** | Stateless (single turn) | Often multi-turn — behavior on turn 3 depends on turns 1–2 |
| **Upstream dependencies** | Prompt input (from preprocessing/RAG) | Every tool the agent calls — APIs, databases, retrieval systems |
| **Migration surface** | One prompt to tune | System prompt + tool descriptions + step prompts + routing logic |

### Two Testing Levels for Agents

> **Note on terminology:** v1 uses "prompt-level" and "pipeline-level" to distinguish isolated-prompt tests from end-to-end tests. For agents, we reuse the same two-level concept but the scope expands: "agent-level" = mocked tools (analogous to prompt-level), "pipeline-level" = real tools (analogous to v1 pipeline-level). Both are controlled by the customer's command, not by a Scaffold config field.

The v1 distinction between prompt-level and pipeline-level testing carries over to agents, and becomes even more important. In v1, the "upstream" is the preprocessing/RAG pipeline feeding the prompt. For agents, every tool is an upstream dependency.

| Level | What it tests | What it catches | Trade-offs |
|---|---|---|---|
| **Agent-level** (unit test) | The agent's decision-making in isolation. Tools are mocked or stubbed — the agent receives canned tool responses. | Changes to: system prompt, tool descriptions, step-level instructions, model, routing logic. | Fast, hermetic, deterministic. Misses changes to tool implementations. |
| **Pipeline-level** (integration test) | The full system with real tools. The agent calls actual APIs, databases, retrieval systems. | Everything — including when a tool's response format changes, an API returns different data, or a database schema is updated. | Slower, depends on services, may require test fixtures. Much more comprehensive. |

**Why pipeline-level matters more for agents than for single calls:** a single LLM call has one upstream dependency (its input). An agent calling 5 tools has 5 upstream dependencies, any of which can change independently. Someone refactors the `lookup_user` API to return `full_name` instead of `first_name` + `last_name`. The agent's prompts are untouched. But now the agent misinterprets the response at step 2 and the whole conversation breaks. An agent-level test with mocked tools would never catch this.

**Implementation:** the same `command` target mode from v1 handles this naturally. The mocked-vs-real distinction is the customer's responsibility, not Scaffold's — the customer controls what their command does internally. Scaffold just evaluates the output trace regardless.

```yaml
evals:
  # Agent-level: customer's command uses mocked tools (fast, hermetic)
  - name: support-agent-unit
    level: agent
    dataset: ./evals/support_scenarios.jsonl
    target:
      command: "python run_agent.py --scenario {input_file} --trace {output_file} --mock-tools"

  # Pipeline-level: customer's command uses real tools (comprehensive)
  - name: support-agent-integration
    level: agent
    dataset: ./evals/support_scenarios.jsonl
    target:
      command: "python run_agent.py --scenario {input_file} --trace {output_file}"
```

The same dataset can be used for both. Scaffold sees `level: agent` in both cases and evaluates identically — the testing-level distinction lives in the customer's command, not in Scaffold's config. This keeps Scaffold's config surface small and avoids a redundant `testing_level` field.

### Eval Dataset for Agents

The JSONL format extends from input/expected pairs to scenarios with outcomes and constraints. Scenarios can be **single-turn** or **multi-turn conversations**.

#### Single-turn scenario

```json
{
  "input": {
    "user_message": "Cancel my subscription and refund the last payment",
    "context": {"user_id": "u_123", "plan": "premium", "last_charge": "$29.99"}
  },
  "expected": {
    "outcome": "Subscription cancelled, $29.99 refund initiated",
    "constraints": {
      "max_tool_calls": 5,
      "required_tools": ["cancel_subscription", "process_refund"],
      "forbidden_tools": ["delete_account"],
      "max_tokens": 30000
    }
  }
}
```

#### Multi-turn conversation scenario

Many agent interactions are multi-turn. A customer asks to cancel, then follows up asking about their remaining balance, then asks to re-subscribe on a different plan. The agent's behavior on turn 3 depends on the full context of turns 1–2.

```json
{
  "turns": [
    {
      "user_message": "Cancel my subscription",
      "context": {"user_id": "u_123", "plan": "premium"},
      "expected": {
        "outcome": "Subscription cancelled, confirmation provided",
        "constraints": {"max_tool_calls": 3, "required_tools": ["cancel_subscription"]}
      }
    },
    {
      "user_message": "Wait, do I get a refund for the remaining days?",
      "expected": {
        "outcome": "Prorated refund amount calculated and communicated",
        "constraints": {"max_tool_calls": 2, "required_tools": ["calculate_prorated_refund"]}
      }
    },
    {
      "user_message": "Ok, actually re-subscribe me but on the basic plan instead",
      "expected": {
        "outcome": "Re-subscribed to basic plan, billing updated",
        "constraints": {
          "max_tool_calls": 3,
          "required_tools": ["create_subscription"],
          "forbidden_tools": ["cancel_subscription"]
        }
      }
    }
  ],
  "conversation_constraints": {
    "max_total_tool_calls": 12,
    "max_total_tokens": 50000
  }
}
```

Key design points for multi-turn:

- **Each turn has its own expected outcome and constraints.** This lets you catch turn-specific regressions (e.g., "the agent handles cancellation fine but breaks on re-subscription follow-ups").
- **`conversation_constraints` apply across the full conversation.** Total tool calls, total tokens, total latency — budgets that matter at the conversation level, not just per-turn.
- **`context` is only on the first turn.** Subsequent turns inherit the conversation state — the agent should remember what happened, not be re-told.
- **Evaluation is per-turn and per-conversation.** llmci evaluates each turn's outcome independently, then computes a conversation-level composite score. A regression on any turn flags the whole scenario.

#### Eval modes for multi-turn

| Mode | How it works | Use case |
|---|---|---|
| **Full replay** | Scaffold invokes the customer's command once per turn, passing the conversation history so far. The agent accumulates real state across turns. | Default for agent-level testing. Catches context management bugs. |
| **History injection** | Scaffold invokes the customer's command once, providing all prior turns as pre-filled chat history. Only the target turn is executed. | Debugging a specific turn. Faster iteration during development. |

The eval config specifies the mode:

```yaml
evals:
  - name: support-conversations
    level: agent
    mode: full_replay          # or "history_injection"
    dataset: ./evals/support_conversations.jsonl
```

### Execution Trace Format

The key new primitive is the **execution trace**. The customer's agent outputs a structured log of every step alongside the final answer. For multi-turn conversations, each turn produces its own trace, nested under the turn index.

#### Single-turn trace

```json
{
  "final_output": "Your subscription has been cancelled and a refund of $29.99 has been initiated.",
  "trace": [
    {"step": 1, "type": "tool_call", "tool": "lookup_user", "args": {"user_id": "u_123"}, "tokens": 1200},
    {"step": 2, "type": "tool_call", "tool": "cancel_subscription", "args": {"user_id": "u_123"}, "tokens": 800},
    {"step": 3, "type": "tool_call", "tool": "process_refund", "args": {"amount": 29.99}, "tokens": 950},
    {"step": 4, "type": "response", "content": "Your subscription has been...", "tokens": 600}
  ],
  "total_tool_calls": 3,
  "total_tokens": 3550
}
```

#### Multi-turn trace

```json
{
  "turns": [
    {
      "turn": 1,
      "output": "Your subscription has been cancelled.",
      "trace": [
        {"step": 1, "type": "tool_call", "tool": "lookup_user", "args": {"user_id": "u_123"}, "tokens": 1200},
        {"step": 2, "type": "tool_call", "tool": "cancel_subscription", "args": {"user_id": "u_123"}, "tokens": 800},
        {"step": 3, "type": "response", "content": "Your subscription has been cancelled.", "tokens": 500}
      ],
      "tool_calls": 2,
      "tokens": 2500
    },
    {
      "turn": 2,
      "output": "You have 18 days remaining. A prorated refund of $17.40 will be issued.",
      "trace": [
        {"step": 1, "type": "tool_call", "tool": "calculate_prorated_refund", "args": {"user_id": "u_123"}, "tokens": 900},
        {"step": 2, "type": "response", "content": "You have 18 days remaining...", "tokens": 600}
      ],
      "tool_calls": 1,
      "tokens": 1500
    }
  ],
  "total_tool_calls": 3,
  "total_tokens": 4000
}
```

llmci doesn't run the agent directly — it invokes the customer's command. Scaffold passes input (and conversation history for multi-turn) to the command, collects the output trace, and evaluates it. In full replay mode, Scaffold calls the command once per turn; in history injection mode, it calls it once with all prior turns pre-filled. Either way, the customer's command owns the agent logic while Scaffold owns the evaluation.

### Eval Config for Agents

```yaml
evals:
  - name: customer-support-agent
    level: agent
    dataset: ./evals/support_scenarios.jsonl
    target:
      command: "python run_agent.py --scenario {input_file} --trace {output_file}"
    judge:
      type: composite
      criteria:
        - name: outcome_correct
          type: llm
          rubric: "Did the agent resolve the customer's issue correctly?"
          weight: 0.5
        - name: trajectory_efficient
          type: llm
          rubric: "Did the agent take a reasonable path without unnecessary steps?"
          weight: 0.2
        - name: tool_budget
          type: constraint
          metric: total_tool_calls
          max: 10
          weight: 0.15
        - name: cost_budget
          type: constraint
          metric: total_tokens
          max: 50000
          weight: 0.15
    metrics:
      - name: composite_score
        threshold: 0.05
        mode: max_regression
```

### Agent-Specific Judges

New judge types beyond what v1 provides:

| Judge | Type | Evaluates |
|---|---|---|
| **Outcome judge** | LLM-as-judge | Did the agent achieve the correct final result? Similar to v1 rubric judge but applied to outcome in context of full scenario. |
| **Constraint judge** | Deterministic | Tool call budget, token budget, latency budget, required/forbidden tools. Fast, cheap, no LLM needed. |
| **Trajectory judge** | LLM-as-judge | Was the execution path efficient and logical? Requires the judge to evaluate the full trace. Hardest to get right. |
| **Composite judge** | Weighted combination | Single score from weighted mix of outcome, constraint, and trajectory judges. |

### Agent Migration

For single-call migration, you tune one prompt. For agent migration, the optimizer must be **trajectory-aware** and modify specific components:

1. Run the agent on the train set with the new model.
2. Identify *where in the trajectory* things go wrong — not just that the final output is wrong, but which step diverged (e.g., wrong tool selected at step 3, misinterpreted tool output at step 5).
3. Diagnose the failure pattern (e.g., "the new model misinterprets the search tool's output format").
4. Suggest targeted modifications to the relevant component — system prompt, a specific tool description, step-level instructions.
5. Re-run, measure on validation set, apply early stopping.

The same core optimization loop (small steps, holdout validation, early stopping) applies. The main difference is that the search space is larger: instead of modifying a single prompt, the optimizer chooses *which* prompt component to modify and how.

### Framework Adapters

Lightweight test-time wrappers that capture execution traces in Scaffold's format when running evals. These run during `llmci run`, not in production — they wrap the agent only during the eval invocation, reducing integration effort from "restructure your agent output" to "wrap your agent in one call":

```python
# OpenAI Agent SDK
from llmci.integrations.openai_agents import traced_agent
agent = traced_agent(existing_agent, trace_output="./traces/")

# PydanticAI
from llmci.integrations.pydantic_ai import traced_agent
agent = traced_agent(existing_agent, trace_output="./traces/")

# Claude Agent SDK
from llmci.integrations.claude_agents import traced_agent
agent = traced_agent(existing_agent, trace_output="./traces/")
```

### v1 → v2 Progression

| | v1 | v2 |
|---|---|---|
| **What it tests** | Single LLM calls | + Agentic workflows (single-turn and multi-turn) |
| **Testing levels** | Prompt-level + pipeline-level | + Agent eval (mocked vs real tools controlled by customer's command) |
| **Eval unit** | Input → output | + Scenario → outcome + trace (per-turn and per-conversation) |
| **Judges** | Exact match, metrics, LLM rubric | + Constraint, trajectory, composite |
| **Migration** | Tune one prompt | + Tune system prompt, tool descriptions, step prompts |
| **Integrations** | Generic (command mode) | + OpenAI Agents, Claude Agent SDK, PydanticAI adapters |
| **Baselines** | Metric scores | + Cost/latency/tool-call budgets |
| **Dataset creation** | Manual curation (`llmci dataset init/add/check`) | + `llmci generate` (log import, synthetic, augmentation) |

The core architecture (config file, dataset format, baseline comparison, CI gate, optimization loop) carries over entirely. The main new investments are the trace format, agent-specific judges, and framework adapters.

### Gold Dataset Creation

Creating eval datasets is the biggest adoption barrier. But it's important to keep the scale in perspective: **these are CI gates, not training sets.** A well-chosen dataset of 100–500 examples is enough to catch regressions with statistical confidence. This isn't big data — it's more like writing unit test fixtures.

#### Strategy 0: Manual Curation (the default)

The simplest and often best approach. A domain expert writes input/expected pairs by hand, focusing on coverage of the important cases rather than volume.

**Why this works for CI gates:**

- You need breadth (cover the categories, edge cases, failure modes), not depth (thousands of examples per category).
- A human who understands the domain writes better edge cases than any synthetic generator.
- 200 carefully chosen examples that cover all failure modes beat 2,000 auto-generated ones that cluster around the easy cases.
- The examples are immediately trustworthy — no review step needed.

**Scaffold should make this easy:**

```bash
# Initialize an empty eval dataset with the right schema
llmci dataset init --name ticket-classification --type deterministic

# Add examples interactively
llmci dataset add --name ticket-classification
# > Input: "My printer won't connect to wifi"
# > Expected: "hardware"
# > Added. (47 examples total, 6 categories covered)

# Validate dataset coverage and quality
llmci dataset check --name ticket-classification
# > 203 examples across 8 categories
# > ⚠ "returns" has only 4 examples (min recommended: 15)
# > ⚠ No multilingual examples detected
# > ✓ All other categories well-covered
```

The `llmci dataset check` command is key — it analyzes the dataset for coverage gaps (underrepresented categories, missing edge case types, lack of diversity) and suggests where to add more examples. This turns manual curation from "write until you feel done" into a guided process with a clear finish line.

For agents, manual curation is harder (you need scenarios, expected outcomes, and constraints), but still viable for a core set of 50–100 scenarios. The automated strategies below are most valuable for expanding beyond this core.

#### Strategy 1: Production Log Import

The most valuable gold data comes from what the system is already doing successfully in production. Most teams already have production logs — from Langfuse, Arize, their own application logging, or exported CSVs. llmci doesn't touch production; it consumes exported logs.

**Scaffold's role is import and curation, not collection.** The flow:

1. Customer exports production logs from whatever system they already use (Langfuse export, Arize export, application database dump, CSV/JSONL files).
2. `llmci generate --from-logs ./exported_logs/` parses the logs into llmci's format. Supports common formats out of the box (Langfuse JSONL, generic JSONL with input/output fields, CSV).
3. Scaffold clusters examples by scenario type and selects diverse, representative cases.
4. An LLM reviews each candidate to identify which examples represent clearly successful outcomes.
5. Scaffold converts successful examples into eval datasets — the input becomes the test case, the output becomes the expected result, and observed metrics (tool calls, tokens) become constraint baselines.
6. Human reviews and approves/edits via `--review`.

```bash
scaffold generate --from-logs ./exported_logs/ \
                  --format langfuse \
                  --output ./evals/agent_scenarios.jsonl \
                  --min-examples 100 \
                  --diversity high
```

This is the highest-quality approach because the data reflects real usage patterns, not synthetic ones. But llmci never runs in production — it only consumes what the team already captures.

#### Strategy 2: Synthetic Scenario Generation

For teams without production logs (new systems, pre-launch), generate scenarios from the system's specification:

1. Customer provides tool/API definitions (OpenAPI specs, function schemas, tool descriptions) and the agent's system prompt.
2. `llmci generate --from-spec` uses an LLM to generate realistic user scenarios that exercise different tool combinations and edge cases.
3. For each scenario, llmci runs the agent to produce a candidate outcome and trace.
4. An LLM (or optionally a human) reviews the output to determine if it's correct, then marks it as gold.

```bash
scaffold generate --from-spec ./agent_config.yaml \
                  --output ./evals/synthetic_scenarios.jsonl \
                  --num-examples 200 \
                  --coverage-targets "all tools used at least 5 times"
```

The `--coverage-targets` flag is important — it ensures the generated scenarios exercise the full tool surface, not just the easy/common paths.

#### Strategy 3: Perturbation-Based Augmentation

Starting from a small seed dataset (even 20-30 examples), generate variations:

1. Take existing eval examples.
2. Use an LLM to create perturbations — rephrase the input, change parameters, add edge cases, combine multiple intents.
3. Run the perturbed inputs through the agent, review outcomes.
4. Add approved perturbations to the dataset.

```bash
scaffold generate --augment ./evals/existing_scenarios.jsonl \
                  --output ./evals/augmented_scenarios.jsonl \
                  --perturbations-per-example 5
```

This is the fastest way to go from a minimal hand-curated set to a statistically meaningful eval dataset.

#### Constraint Inference

A unique challenge for agent eval datasets is setting reasonable constraints. What's a fair `max_tool_calls` or `max_tokens` for a given scenario? llmci can infer these automatically:

1. Run each scenario N times (e.g., 10) on the current agent locally.
2. Compute percentile-based bounds from observed behavior (e.g., p90 tool calls + 20% buffer).
3. Set these as the default constraints in the generated dataset.

This avoids the problem of humans guessing at arbitrary constraint values and produces bounds grounded in actual system behavior.

#### The Review Interface

Generated datasets always require human review. Scaffold should output a review-friendly format:

```bash
scaffold generate --from-logs ./exported_logs/ --output ./evals/draft.jsonl --review
```

The `--review` flag opens an interactive CLI session where the reviewer sees each candidate example and can approve, edit, or reject it:

```
Example 14 of 87 — Source: exported log 2024-03-15T14:22:01Z

Scenario: User asks to cancel subscription and get a refund
Input: {"user_message": "Cancel my sub and refund me", "context": {...}}
Outcome: "Subscription cancelled, $29.99 refund initiated"
Constraints: max_tool_calls=5, required_tools=[cancel_subscription, process_refund]
Trace: 3 tool calls, 3550 tokens

[a]pprove  [e]dit  [r]eject  [s]kip  >
```

#### CLI Surface

```bash
# Import from exported production logs (Langfuse, Arize, generic JSONL/CSV)
scaffold generate --from-logs <log_dir> --format <langfuse|jsonl|csv> --output <output.jsonl>

# Generate synthetic scenarios from agent spec
scaffold generate --from-spec <agent_config> --output <output.jsonl>

# Augment an existing small dataset
scaffold generate --augment <existing.jsonl> --output <output.jsonl>

# Infer constraints from observed agent behavior
scaffold generate --infer-constraints <dataset.jsonl> --runs 10
```

### Competitive Angle

Nobody is doing CI-gated regression testing for agentic systems today. Teams deploy agent changes and hope for the best. The complexity and non-determinism of agents makes a safety gate *more* valuable, not less — this is where Scaffold's positioning gets strongest.

The dataset generation feature further widens the moat. Competitors assume you already have eval data. Scaffold helps you create it — meeting teams where they actually are, not where you wish they were.

---

## Future Considerations (beyond v2)

### Red Teaming / Security Scanning

Promptfoo's strongest feature is its red-teaming suite (142+ attack plugins, OWASP LLM Top 10 coverage). DeepEval has similar vulnerability scanning. Scaffold v1-v2 intentionally omits this — regression testing and migration are the core mission. However, security scanning is a natural extension:

- `llmci scan` could run adversarial inputs (prompt injection, jailbreaks, PII leakage) against the target and fail the CI check if vulnerabilities are found.
- This would be a v3 feature or a separate companion tool. Adding it too early dilutes the "focused CI gate" positioning.

### RAGAS Integration as a Judge Type

RAGAS is the best-in-class RAG evaluation library (retrieval quality, context relevance, faithfulness). Rather than reimplementing these metrics, Scaffold could integrate RAGAS as a pluggable judge:

```yaml
judge:
  type: ragas
  metrics: [faithfulness, context_relevancy, answer_relevancy]
```

This gives Scaffold RAG-specific evaluation for free while maintaining focus on the CI gating infrastructure.

### Ecosystem Positioning

llmci is deliberately composable. It does one thing (CI quality gate) and does it well. The expected stack for a mature LLM team:

| Layer | Tool |
|---|---|
| **Production monitoring** | Langfuse, Arize Phoenix, Braintrust |
| **Pre-merge testing** | **Scaffold** |
| **Prompt development** | DSPy, manual iteration |
| **Security scanning** | Promptfoo (or future `llmci scan`) |

This positioning avoids platform sprawl and makes adoption frictionless — Scaffold slots in alongside whatever observability tool the team already uses.

---

## Open Questions

- [ ] PyPI package name availability (`llmci`, `evalgate`, etc.)
- [ ] Stretch goal (v1): `llmci init` auto-detecting prompt files and suggesting eval structure
- [ ] Whether to support a remote baseline service in v1 or defer to paid tier
- [ ] Promptfoo config import: how faithful should `llmci import-promptfoo` be? Full parity or just the most common patterns?
- [ ] Should Scaffold support a `--watch` mode for local development (re-run evals on file save) to capture some of the dev-time usage that Promptfoo handles?
- [ ] v2: Standard trace format — adopt an existing standard (e.g., OpenTelemetry spans) or define a custom one?
- [ ] v2: How to handle agent non-determinism in baselines — run N times and average, or use outcome-only baselines?
- [ ] v2: Scope of framework adapters — which SDKs to support first based on adoption?
- [ ] v3: Red teaming — build in-house or integrate with existing tools?
- [ ] Cost estimation: should `llmci run --dry-run` or `--estimate-cost` preview the expected API spend before running? Useful for large datasets or expensive judge models.
- [ ] Community: should Scaffold have a public eval dataset registry (like a package registry but for gold datasets)?
