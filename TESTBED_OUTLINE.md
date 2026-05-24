# Scaffold Testbed — Implementation Outline

This document specifies a **separate GitHub repository** (`scaffold-testbed`) that acts as a realistic customer monorepo for dogfooding [Scaffold](https://github.com/alexminnaar/scaffold) (`llmci` on PyPI). An implementer should be able to build the entire repo from this outline without reading Scaffold's source code.

**Related docs in the llmci repo:**
- Case studies: `docs.html` (sections `cs-fastapi`, `cs-rag`, `cs-migration`, `cs-agent`, `cs-summarization`)
- Minimal feature examples: `examples/01`–`09`
- llmci CLI reference: `docs.html#cli-ref`

---

## 1. Purpose

| Goal | Detail |
|------|--------|
| Realistic adoption story | Multi-file services, HTTP APIs, pre/post processing — not 50-line scripts |
| Case study coverage | Every docs case study has a runnable service + eval config |
| Agent coverage | Single-turn, multi-turn (full replay), optional history injection |
| CI without API cost | Default `MOCK_LLM=1`; deterministic passes in GitHub Actions |
| PR demo | Intentional regression branches show Scaffold PR comments failing |
| PyPI dependency | Install `llmci` from PyPI (or git `@main` during active dev) |

**Not in scope:** production deployment, real auth, observability, hosted Scaffold.

---

## 2. Repository metadata

| Field | Recommendation |
|-------|----------------|
| GitHub org/repo | `<org>/scaffold-testbed` (same org as main Scaffold project) |
| License | Apache 2.0 (match Scaffold) |
| Python | 3.11+ |
| Primary dependency | `llmci>=0.1.0` |
| Branding | Fictional company **Acme Support** — ticket classifier, RAG helpdesk, support agent |

---

## 3. Top-level structure

```
scaffold-testbed/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml                 # service deps + llmci
├── Makefile                       # convenience targets (see §12)
├── docker-compose.yml             # optional: HTTP services for local dev
│
├── .github/
│   └── workflows/
│       ├── scaffold.yml           # PR CI — mock mode, matrix of services
│       └── scaffold-llm.yml       # manual — real LLM (secrets required)
│
├── shared/
│   ├── README.md
│   ├── mock_llm.py                # deterministic LLM stub
│   └── scripts/
│       └── wait_for_http.sh       # curl loop for CI service startup
│
├── services/
│   ├── ticket-classifier/         # §5 — FastAPI case study
│   ├── rag-qa/                    # §6 — RAG pipeline case study
│   ├── summarizer/                # §7 — Summarization QA case study
│   ├── support-agent/             # §8 — Agent case study
│   └── json-api/                  # §9 — Custom judge / structured output
│
└── migration/                     # §10 — Model migration case study
```

Each service directory is **self-contained**: own `llmci.yaml`, `evals/`, README, and CI `working-directory`.

---

## 4. Shared contracts

### 4.1 Standard command-mode I/O (prompt / pipeline evals)

Scaffold writes an **input file** and expects an **output file** with JSON:

**Input** (written by llmci):
```json
{"input": "user text here", "expected": "optional gold label"}
```

**Output** (written by your script):
```json
{"output": "model or pipeline response"}
```

CLI pattern in `llmci.yaml`:
```yaml
target:
  command: "python3 scripts/run.py --input {input_file} --output {output_file}"
```

Use `python3` (not `python`) for Linux CI compatibility.

### 4.2 Agent command-mode I/O

**Single-turn input:**
```json
{"query": "What's the weather in New York?"}
```
or `{"input": {"query": "..."}}` — agent runner normalizes dict inputs.

**Multi-turn input (full replay, one invocation per turn):**
```json
{
  "user_message": "Can you cancel it?",
  "history": [
    {"role": "user", "content": "What's the status of my order?"},
    {"role": "assistant", "content": "Order #1234 is shipped."}
  ],
  "turn_index": 1
}
```

**Agent output** (required fields):
```json
{
  "final_output": "Your order has been cancelled.",
  "trace": [
    {"step": 1, "type": "tool_call", "tool": "lookup_order", "args": {"id": "1234"}, "content": "...", "tokens": 20},
    {"step": 2, "type": "response", "content": "Your order has been cancelled.", "tokens": 35}
  ],
  "total_tool_calls": 1,
  "total_tokens": 55
}
```

Scaffold also accepts `"output"` as alias for `final_output`.

### 4.3 JSONL dataset — standard eval

One JSON object per line:
```json
{"input": "My printer keeps jamming", "expected": "hardware"}
```

Minimum **20 rows** per classification dataset; **5+ rows** for smaller demos.

### 4.4 JSONL dataset — agent single-turn

```json
{
  "input": {"query": "I want to return order #5678"},
  "expected": {
    "outcome": "return initiated",
    "constraints": {
      "required_tools": ["lookup_order", "initiate_return"],
      "forbidden_tools": ["delete_account", "issue_refund"],
      "max_tool_calls": 4
    }
  }
}
```

### 4.5 JSONL dataset — agent multi-turn

```json
{
  "turns": [
    {
      "user_message": "I want to return order #5678",
      "expected": {
        "outcome": "return initiated",
        "constraints": {"required_tools": ["lookup_order", "initiate_return"], "max_tool_calls": 4}
      }
    },
    {
      "user_message": "Actually, can I get a refund instead?",
      "expected": {
        "outcome": "refund processed",
        "constraints": {"required_tools": ["issue_refund"]}
      }
    }
  ],
  "conversation_constraints": {"max_tool_calls": 6, "max_tokens": 800}
}
```

### 4.6 Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MOCK_LLM` | `1` in CI | Use deterministic keyword/fixture logic instead of real API calls |
| `OPENAI_API_KEY` | unset in CI | Required for real LLM judge / migration / direct target |
| `CLASSIFIER_URL` | `http://localhost:8000` | HTTP eval wrapper for ticket-classifier |
| `GITHUB_TOKEN` | auto in Actions | PR comment posting |

### 4.7 `shared/mock_llm.py`

Provide a tiny module importable by all services:

```python
def is_mock() -> bool:
    return os.environ.get("MOCK_LLM", "0") == "1"

def complete(prompt: str, *, model: str = "mock") -> str:
    """Return deterministic text from prompt keywords. Used when MOCK_LLM=1."""
```

When `MOCK_LLM=0`, services call `litellm` or OpenAI SDK with real credentials.

**CI strategy for LLM-as-judge evals (summarizer):**
- **Option A (recommended):** In mock CI, use `judge: custom` with a lightweight heuristic judge; keep `judge: llm` config in `scaffold-llm.yaml` for manual runs.
- **Option B:** Run summarizer only in `scaffold-llm.yml` workflow, exclude from default matrix.

---

## 5. Service: `ticket-classifier`

**Maps to:** docs case study `cs-fastapi`, example `08-fastapi-service`.

### 5.1 Purpose

FastAPI microservice that classifies support tickets. Includes **pre-processing** (normalize, PII redaction) and **post-processing** (confidence threshold → route to `general`). Two eval levels:
- **Prompt-level:** classification logic only
- **Service-level:** full pipeline via HTTP or in-process wrapper

### 5.2 Directory layout

```
services/ticket-classifier/
├── README.md
├── pyproject.toml                 # optional: fastapi, uvicorn — or use root pyproject
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app, POST /classify
│   ├── pipeline.py                # preprocess → classify → postprocess
│   ├── classifier.py              # LLM or mock keyword classifier
│   └── prompts/
│       └── classify.txt
├── evals/
│   └── tickets.jsonl              # ≥24 rows, 4 categories
├── scripts/
│   ├── run_prompt.py              # prompt-level wrapper (classify only)
│   └── eval_service.py            # service-level wrapper (full pipeline)
├── llmci.yaml                  # service-level eval (default)
├── scaffold-prompt.yaml           # prompt-level eval
└── .llmci/
    └── baselines/                 # committed after first `--update-baseline` on main
```

### 5.3 HTTP API

**`POST /classify`**
```json
// Request
{"text": "I was charged twice for my subscription"}

// Response
{"category": "billing", "confidence": 3, "preprocessed_text": "I was charged twice..."}
```

**`GET /health`** → `{"status": "ok"}`

### 5.4 Classifier behavior

| Mode | Behavior |
|------|----------|
| `MOCK_LLM=1` | Keyword scoring (hardware/billing/account/software) — copy logic from `scaffold/examples/08-fastapi-service/service.py` |
| `MOCK_LLM=0` | Load `prompts/classify.txt`, call LLM via litellm with `{input}` substitution |

**Post-processing:** if keyword score `< CONFIDENCE_THRESHOLD` (default `2`), return category `general`.

**PII redaction:** phone, email, credit card, SSN patterns → `[PHONE]`, `[EMAIL]`, etc.

### 5.5 `llmci.yaml` (service-level)

```yaml
version: 1

target:
  command: "python3 scripts/eval_service.py --input {input_file} --output {output_file}"

evals:
  - name: service-classification
    level: pipeline
    dataset: ./evals/tickets.jsonl
    judge: exact_match
    metrics:
      - name: accuracy
        threshold: 0.75
        mode: absolute
      - name: f1_weighted
        threshold: 0.70
        mode: absolute
      - name: accuracy
        threshold: 0.03
        mode: max_regression

settings:
  parallelism: 5
  timeout_per_call: 15
  retries: 1
```

### 5.6 `scaffold-prompt.yaml` (prompt-level)

```yaml
version: 1

target:
  command: "python3 scripts/run_prompt.py --input {input_file} --output {output_file}"

evals:
  - name: prompt-classification
    level: prompt
    dataset: ./evals/tickets.jsonl
    judge: exact_match
    metrics:
      - name: accuracy
        threshold: 0.90
        mode: absolute
      - name: f1_macro
        threshold: 0.85
        mode: absolute
```

### 5.7 `scripts/eval_service.py`

- Read input JSON `{"input": "...", "expected": "..."}`
- Call `pipeline.classify(text)` in-process (preferred for CI) **or** HTTP POST to `CLASSIFIER_URL`
- Write `{"output": "<category>"}`

### 5.8 Acceptance criteria

- [ ] `cd services/ticket-classifier && MOCK_LLM=1 llmci run` exits 0 on `main`
- [ ] `MOCK_LLM=1 llmci run --config scaffold-prompt.yaml` exits 0 on `main`
- [ ] `uvicorn app.main:app --port 8000` starts; `/health` returns 200
- [ ] Branch `test/break-classifier`: remove billing keywords → service-level accuracy fails → PR comment posted

---

## 6. Service: `rag-qa`

**Maps to:** docs case study `cs-rag`, example `07-pipeline-level`.

### 6.1 Purpose

Mock RAG pipeline: retrieve from in-memory knowledge base → generate answer. Tests **pipeline-level** regressions (retrieval + generation), not prompt alone.

### 6.2 Directory layout

```
services/rag-qa/
├── README.md
├── pipeline/
│   ├── __init__.py
│   ├── retrieve.py
│   ├── generate.py
│   └── run.py                     # CLI entry: --input / --output
├── evals/
│   └── qa.jsonl                   # ≥10 Q&A pairs
├── rag_judge.py                   # custom judge module
├── llmci.yaml
└── .llmci/baselines/
```

### 6.3 Dataset format

```json
{"input": "What is Docker used for?", "expected": "containerization;images;containers"}
```

`expected` is semicolon-separated **required fact substrings** (not exact match).

### 6.4 Custom judge (`rag_judge.py`)

```python
def evaluate(input: str, expected: str, actual: str) -> dict:
    facts = [f.strip() for f in expected.split(";") if f.strip()]
    found = sum(1 for f in facts if f.lower() in actual.lower())
    score = found / len(facts) if facts else 1.0
    return {"score": score, "reason": f"{found}/{len(facts)} facts found"}
```

### 6.5 `llmci.yaml`

```yaml
version: 1

target:
  command: "python3 pipeline/run.py --input {input_file} --output {output_file}"

evals:
  - name: rag-qa
    level: pipeline
    dataset: ./evals/qa.jsonl
    judge:
      type: custom
      module: ./rag_judge.py
      function: evaluate
    metrics:
      - name: pass_rate
        threshold: 0.80
        mode: absolute
      - name: mean_score
        threshold: 0.75
        mode: absolute

settings:
  parallelism: 5
  timeout_per_call: 10
  retries: 1
```

### 6.6 Knowledge base

Embed a small static dict (Python, Docker, Git, Kubernetes topics) — copy from `examples/07-pipeline-level/run_pipeline.py`. Retrieval = keyword overlap; generation = template wrapping context.

### 6.7 Acceptance criteria

- [ ] `MOCK_LLM=1 llmci run` exits 0
- [ ] `test/break-rag-retrieval`: break retrieval (e.g. `top_k=0`) → pass_rate fails

---

## 7. Service: `summarizer`

**Maps to:** docs case study `cs-summarization`, example `09-summarization-qa`.

### 7.1 Purpose

Article → summary pipeline. Two evals: with reference summaries and reference-free. Demonstrates LLM-as-judge with multi-criteria rubrics.

### 7.2 Directory layout

```
services/summarizer/
├── README.md
├── app/
│   ├── summarizer.py              # extractive or mock LLM summarizer
│   └── run.py                     # command wrapper
├── evals/
│   ├── articles_with_refs.jsonl   # ≥6 rows with "expected"
│   └── articles_no_refs.jsonl     # ≥6 rows, input only
├── llmci.yaml                  # both evals (LLM judge)
├── scaffold-mock.yaml             # CI: custom heuristic judge OR skip
└── README.md
```

### 7.3 `llmci.yaml` (real LLM judge)

Copy structure from `examples/09-summarization-qa/llmci.yaml`:
- Eval `summary-with-refs`: metrics `mean_score`, `min_score`, `cosine_similarity`
- Eval `summary-no-refs`: metrics `mean_score`, `pass_rate`
- Rubrics: `faithfulness`, `completeness`, `conciseness`
- Judge model: `openai/gpt-4o-mini`

### 7.4 `scaffold-mock.yaml` (CI)

Replace judge with custom module that scores extractive overlap (for deterministic CI):

```yaml
judge:
  type: custom
  module: ./mock_summary_judge.py
  function: evaluate
```

### 7.5 Summarizer implementation

- `MOCK_LLM=1`: extractive summarizer (keyword density + lead bias) from `examples/09-summarization-qa/run_summarizer.py`
- `MOCK_LLM=0`: optional real LLM call

### 7.6 Acceptance criteria

- [ ] `MOCK_LLM=1 llmci run --config scaffold-mock.yaml` exits 0 in CI
- [ ] Manual: `OPENAI_API_KEY=... llmci run` passes with real judge (document in README)

---

## 8. Service: `support-agent`

**Maps to:** docs case study `cs-agent`, examples `05-agent-single-turn`, `06-agent-multi-turn`.

### 8.1 Purpose

Mock customer-support agent with tools: `lookup_order`, `cancel_order`, `issue_refund`, `search_kb`, `initiate_return`, etc. Composite judge with constraint / outcome / trajectory weights.

### 8.2 Directory layout

```
services/support-agent/
├── README.md
├── agent/
│   ├── __init__.py
│   ├── tools.py                   # mock tool implementations
│   └── run_agent.py               # CLI: --input / --output
├── evals/
│   ├── scenarios.jsonl            # ≥8 single-turn scenarios
│   └── conversations.jsonl        # ≥4 multi-turn conversations
├── scaffold-single.yaml
├── scaffold-multi.yaml
├── scaffold-history.yaml          # optional stretch: history_injection
└── .llmci/baselines/
```

### 8.3 Agent tools (mock)

| Tool | Purpose |
|------|---------|
| `lookup_order` | Return order status by ID |
| `initiate_return` | Start return flow |
| `issue_refund` | Process refund |
| `search_kb` | Search help articles |
| `delete_account` | **Forbidden** in most scenarios |

Implement keyword routing in `run_agent.py` similar to `examples/05-agent-single-turn/run_agent.py`, extended for support domain.

### 8.4 `scaffold-single.yaml`

```yaml
version: 1

target:
  command: "python3 agent/run_agent.py --input {input_file} --output {output_file}"

evals:
  - name: support-agent-single
    level: agent
    dataset: ./evals/scenarios.jsonl
    judge:
      type: composite
      model: openai/gpt-4o-mini    # for outcome/trajectory; constraint is deterministic
      criteria:
        - name: safety
          type: constraint
          weight: 3.0
        - name: correctness
          type: outcome
          weight: 2.0
        - name: efficiency
          type: trajectory
          weight: 1.0
          rubric: "Did the agent resolve the issue in a reasonable number of steps without redundant tool calls?"
    metrics:
      - name: mean_score
        threshold: 0.75
        mode: absolute
      - name: error_rate
        threshold: 0.05
        mode: absolute

settings:
  parallelism: 3
  timeout_per_call: 20
```

**CI note:** Outcome/trajectory criteria call LLM. For mock CI either:
- Set low weights on LLM criteria and rely on constraint judge, **or**
- Use `MOCK_LLM=1` + patch/fixture responses for composite LLM sub-judges, **or**
- Run full composite only in `scaffold-llm.yml`

**Pragmatic CI config:** duplicate eval with only `type: constraint` judge (like examples 05/06) for default matrix; keep full composite in `scaffold-single-full.yaml` for manual runs.

### 8.5 `scaffold-multi.yaml`

```yaml
evals:
  - name: support-agent-multi
    level: agent
    mode: full_replay
    dataset: ./evals/conversations.jsonl
    judge:
      type: composite
      # same criteria as single-turn
```

### 8.6 `scaffold-history.yaml` (stretch)

```yaml
evals:
  - name: support-agent-history
    level: agent
    mode: history_injection
    dataset: ./evals/conversations.jsonl
    # ...
```

Agent must handle input shape from §4.2 history injection.

### 8.7 Acceptance criteria

- [ ] Single-turn constraint-only CI config passes on `main`
- [ ] Multi-turn full replay runs without agent errors
- [ ] `test/break-agent-safety`: agent calls `delete_account` → constraint judge fails

---

## 9. Service: `json-api`

**Maps to:** example `04-custom-judge`.

### 9.1 Purpose

API returns structured JSON; custom judge validates schema + required fields.

### 9.2 Layout

```
services/json-api/
├── README.md
├── api/
│   └── run_api.py
├── evals/
│   └── api_responses.jsonl
├── json_judge.py
└── llmci.yaml
```

Copy from `examples/04-custom-judge/`. Judge function validates JSON parse + required keys.

### 9.3 Acceptance criteria

- [ ] `MOCK_LLM=1 llmci run` exits 0 with `accuracy: 1.0`

---

## 10. Module: `migration`

**Maps to:** docs case study `cs-migration`, example `02-model-migration`.

### 10.1 Purpose

Demonstrate `llmci migrate --from ... --to ...` on ticket classification prompt.

### 10.2 Layout

```
migration/
├── README.md
├── prompts/
│   └── classify.txt               # symlink or copy from ticket-classifier
├── evals/
│   └── tickets.jsonl              # same dataset as classifier (≥20 rows)
└── llmci.yaml
```

### 10.3 `llmci.yaml`

```yaml
version: 1

target:
  provider: openai
  model: gpt-4o-mini
  prompt_file: ./prompts/classify.txt

evals:
  - name: ticket-classification
    level: prompt
    dataset: ./evals/tickets.jsonl
    judge: exact_match
    metrics:
      - name: accuracy
        threshold: 0.85
        mode: absolute

settings:
  parallelism: 3
  timeout_per_call: 30
  retries: 1
```

### 10.4 Usage (document in README)

```bash
export OPENAI_API_KEY=...
llmci migrate \
  --from openai/gpt-4o \
  --to openai/gpt-4o-mini \
  --eval ticket-classification \
  --patience 3 \
  --max-iterations 10
```

### 10.5 CI

**Exclude from default PR matrix** (requires API key + costs). Include in `scaffold-llm.yml` only.

### 10.6 Acceptance criteria

- [ ] Manual migration run completes and writes optimized prompt
- [ ] Document expected holdout score range in README

---

## 11. Root `pyproject.toml`

```toml
[project]
name = "scaffold-testbed"
version = "0.1.0"
description = "Realistic services for dogfooding Scaffold"
requires-python = ">=3.11"
dependencies = [
  "llmci>=0.1.0",
  "fastapi>=0.110",
  "uvicorn>=0.27",
  "httpx>=0.27",
  "litellm>=1.0",
  "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "ruff>=0.4",
]

[tool.ruff]
line-length = 100
target-version = "py311"
```

Install in CI:
```bash
pip install -e ".[dev]"
pip install --upgrade llmci   # or pin: llmci==0.1.0
```

**During active Scaffold development:**
```bash
pip install "llmci @ git+https://github.com/<org>/scaffold@main"
```

---

## 12. `Makefile` targets

```makefile
.PHONY: install eval-all eval-classifier eval-agent test

install:
	pip install -e ".[dev]"

eval-classifier:
	cd services/ticket-classifier && MOCK_LLM=1 llmci run

eval-classifier-prompt:
	cd services/ticket-classifier && MOCK_LLM=1 llmci run --config scaffold-prompt.yaml

eval-all:
	$(MAKE) eval-classifier
	cd services/rag-qa && MOCK_LLM=1 llmci run
	cd services/json-api && MOCK_LLM=1 llmci run
	cd services/support-agent && MOCK_LLM=1 llmci run --config scaffold-single.yaml
	cd services/support-agent && MOCK_LLM=1 llmci run --config scaffold-multi.yaml

test:
	pytest shared/ services/ -q
```

---

## 13. GitHub Actions

### 13.1 `.github/workflows/scaffold.yml`

```yaml
name: llmci Evals

on:
  pull_request:
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: ruff check services/ shared/

  eval:
    runs-on: ubuntu-latest
    needs: [lint]
    permissions:
      contents: read
      pull-requests: write
    strategy:
      fail-fast: false
      matrix:
        include:
          - service: ticket-classifier
            config: llmci.yaml
          - service: ticket-classifier
            config: scaffold-prompt.yaml
          - service: rag-qa
            config: llmci.yaml
          - service: json-api
            config: llmci.yaml
          - service: support-agent
            config: scaffold-single.yaml
          - service: support-agent
            config: scaffold-multi.yaml
          - service: summarizer
            config: scaffold-mock.yaml
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]" && pip install --upgrade llmci
      - name: Run eval
        working-directory: services/${{ matrix.service }}
        env:
          MOCK_LLM: "1"
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: llmci run --config ${{ matrix.config }}
```

**Important:** Do **not** set workflow-level `permissions: pull-requests: write` without `contents: read` — that breaks checkout (learned from Scaffold dogfood).

### 13.2 `.github/workflows/scaffold-llm.yml`

```yaml
name: Scaffold LLM Evals

on:
  workflow_dispatch:
    inputs:
      service:
        description: "Service directory name"
        required: true
        type: choice
        options:
          - summarizer
          - migration
          - support-agent

jobs:
  eval:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]" && pip install --upgrade llmci
      - working-directory: services/${{ inputs.service }}
        env:
          MOCK_LLM: "0"
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: llmci run
```

For `migration`, set `working-directory: migration`.

---

## 14. Demo regression branches

Create from `main` after all services pass:

| Branch | Change | Expected failure |
|--------|--------|------------------|
| `test/break-classifier` | Remove billing keywords in `classifier.py` | `ticket-classifier` accuracy |
| `test/break-rag-retrieval` | Force empty retrieval | `rag-qa` pass_rate |
| `test/break-agent-safety` | Agent invokes forbidden tool | `support-agent` constraint score |

Each branch should produce a **Scaffold PR comment** with failed examples table (verify `GITHUB_TOKEN` + `pull-requests: write` on eval job only).

---

## 15. `docker-compose.yml` (optional)

```yaml
services:
  ticket-classifier:
    build: ./services/ticket-classifier
    ports: ["8000:8000"]
    environment:
      MOCK_LLM: "1"
```

Use for local HTTP testing of `eval_service.py` with `CLASSIFIER_URL=http://localhost:8000`.

---

## 16. Root `README.md` outline

1. What this repo is (Scaffold customer-zero testbed)
2. Prerequisites: Python 3.11+, `pip install -e .`
3. Quick start: `make eval-classifier`
4. Table linking each service → docs case study → Scaffold example
5. CI badge
6. Mock vs real LLM modes
7. Link to main Scaffold repo + PyPI `llmci`
8. How to add a new service

---

## 17. Implementation phases

Implement in this order. **Do not start phase N+1 until phase N acceptance criteria pass.**

### Phase 1 — Repo skeleton (Day 1)
- [ ] Create repo, LICENSE, `.gitignore`, root `pyproject.toml`, `shared/mock_llm.py`
- [ ] `scaffold.yml` with one placeholder job
- [ ] Root README stub

### Phase 2 — Ticket classifier (Days 2–3)
- [ ] Full §5 implementation
- [ ] CI matrix entries for both scaffold configs
- [ ] Baselines on `main`: `llmci run --update-baseline`
- [ ] `test/break-classifier` branch

### Phase 3 — JSON API + RAG (Days 4–5)
- [ ] §9 and §6
- [ ] CI matrix entries

### Phase 4 — Support agent (Days 6–8)
- [ ] §8 single + multi-turn
- [ ] Constraint-only CI configs first; composite manual workflow second
- [ ] `test/break-agent-safety` branch

### Phase 5 — Summarizer mock + LLM workflow (Day 9)
- [ ] §7 with `scaffold-mock.yaml` in CI
- [ ] `scaffold-llm.yml` for real judge

### Phase 6 — Migration module (Day 10)
- [ ] §10, manual docs only

### Phase 7 — Polish (Day 11+)
- [ ] docker-compose
- [ ] Link from Scaffold `docs.html` and `README.md`
- [ ] Optional: `scaffold-history.yaml`

---

## 18. Linking from Scaffold main repo

After testbed is public, update Scaffold repo:

| File | Change |
|------|--------|
| `docs.html` case studies | Add "Full service example → [scaffold-testbed/services/...](url)" |
| `README.md` | Section "Reference integration" linking to testbed |
| `TESTBED_OUTLINE.md` | Keep in Scaffold repo as spec; testbed README points here |

---

## 19. Case study coverage checklist

| Docs case study | Testbed path | CI (mock) | CI (LLM) |
|-----------------|--------------|-----------|----------|
| FastAPI Service | `services/ticket-classifier/` | ✅ both levels | optional |
| RAG Pipeline | `services/rag-qa/` | ✅ | — |
| Summarization QA | `services/summarizer/` | ✅ mock judge | ✅ workflow |
| Support Agent | `services/support-agent/` | ✅ constraint | ✅ composite |
| Multi-Model Migration | `migration/` | — | ✅ workflow |
| Custom judge (extra) | `services/json-api/` | ✅ | — |

---

## 20. Open decisions (fill before implementation)

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Org name | `scaffold-ai` (GitHub), personal | Create org early |
| PyPI package | `llmci` | Short for “LLM CI”; CLI remains `scaffold` |
| PyPI pin | `>=0.1.0` in pyproject; CI `--upgrade llmci` | PyPI is live; git `@main` only for llmci dev |
| Summarizer CI judge | mock custom vs skip | mock custom (`scaffold-mock.yaml`) |
| Agent composite in CI | full vs constraint-only | constraint-only default matrix |
| Baselines | commit `.llmci/baselines/` | yes, on `main` after first green run |

---

## Appendix A — Copy sources in Scaffold repo

| Testbed component | Copy/adapt from |
|-------------------|-----------------|
| Ticket classifier logic | `examples/08-fastapi-service/` |
| RAG pipeline | `examples/07-pipeline-level/` |
| Summarizer | `examples/09-summarization-qa/` |
| Agent single-turn | `examples/05-agent-single-turn/` |
| Agent multi-turn | `examples/06-agent-multi-turn/` |
| JSON judge | `examples/04-custom-judge/` |
| Migration config | `examples/02-model-migration/` |
| CI permissions pattern | `.github/workflows/scaffold-dogfood.yml` |

## Appendix B — llmci CLI quick reference

```bash
llmci run                              # run evals in cwd
llmci run --config scaffold-prompt.yaml
llmci run --compare-to=main            # baseline regression
llmci run --update-baseline            # save baselines (main branch)
llmci migrate --from X --to Y --eval NAME
llmci dataset check ./evals/foo.jsonl
```

---

*Document version: 1.0 — matches Scaffold phases 0–7 and docs case studies as of 2026-05.*
