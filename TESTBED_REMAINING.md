# Scaffold Testbed — Remaining Work

Handoff document for completing [`scaffold-testbed`](https://github.com/alexminnaar/scaffold-testbed) after the initial implementation. The testbed repo lives at `/Users/alexminnaar/projects/scaffold-testbed` locally.

**Spec (what to build):** [TESTBED_OUTLINE.md](./TESTBED_OUTLINE.md)  
**Status (May 2026):** ~85% complete — all services, CI workflows, baselines, and mock evals exist. Remaining work is mostly **operational** (git, GitHub, regression CI) and **polish** (docs, demo branches, small code fixes).

**PyPI:** Testbed should install [`llmci`](https://pypi.org/project/llmci/) from PyPI (`llmci>=0.1.0` in `pyproject.toml`). CI already runs `pip install --upgrade llmci`.

---

## 1. Current state (done)

| Area | Status |
|------|--------|
| Services | `ticket-classifier`, `rag-qa`, `summarizer`, `json-api`, `support-agent`, `migration/` |
| Shared | `mock_llm.py`, `scaffold_run.sh`, `wait_for_http.sh` |
| CI | `.github/workflows/scaffold.yml` (mock matrix), `scaffold-llm.yml` (manual LLM) |
| Baselines | Committed per-service under `.llmci/baselines/` |
| Tests | `make test` — 4 passed (`shared/`, ticket-classifier pipeline) |
| Local evals | `make eval-all` passes in mock mode (with `scaffold_run.sh` for alt configs) |
| PyPI dep | `llmci>=0.1.0`; README updated for PyPI-first install |

---

## 2. P0 — Unblock core purpose

These must happen before the testbed can dogfood Scaffold end-to-end (PR comments, regression detection, demo branches).

### 2.1 Initialize git and push to GitHub

**Problem:** Repo is not a git repository. Baselines have `"commit_sha": "unknown"`. `--compare-to` cannot load baselines from a git ref.

**Tasks:**

```bash
cd scaffold-testbed
git init
git add .
git commit -m "Initial Acme Support testbed"
git remote add origin git@github.com:<org>/scaffold-testbed.git
git branch -M main
git push -u origin main
```

**Acceptance:**
- [ ] Repo public (or org-visible) on GitHub
- [ ] GitHub Actions runs on push/PR
- [ ] Re-run baselines on `main` with real commit SHA:
  ```bash
  cd services/ticket-classifier && MOCK_LLM=1 llmci run --update-baseline
  # repeat for each service / config that has baselines
  git add .llmci/baselines/ && git commit -m "Refresh baselines with commit SHA"
  ```

### 2.2 Add `--compare-to=origin/main` in CI

**Problem:** Ticket classifier defines `max_regression` on accuracy, but plain `llmci run` skips regression checks. Scaffold loads regression baselines from a **git ref**, not the local filesystem.

**File:** `.github/workflows/scaffold.yml` — eval job run step

**Change** `shared/scripts/scaffold_run.sh` invocation to pass through extra args, or inline:

```yaml
run: |
  ../../shared/scripts/scaffold_run.sh --config ${{ matrix.config }} --compare-to=origin/main
```

Ensure checkout uses sufficient history:

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0
```

**Optional — refresh baselines on main push:**

```yaml
- name: Run eval
  run: |
    ARGS="--config ${{ matrix.config }}"
    if [ "${{ github.event_name }}" = "push" ] && [ "${{ github.ref }}" = "refs/heads/main" ]; then
      ARGS="$ARGS --update-baseline"
    else
      ARGS="$ARGS --compare-to=origin/main"
    fi
    ../../shared/scripts/scaffold_run.sh $ARGS
```

**Acceptance:**
- [ ] PR runs show regression comparison (no "baseline not found" skip for `max_regression`)
- [ ] `GITHUB_TOKEN` + `pull-requests: write` posts Scaffold PR comment on PRs
- [ ] Green `main` with committed baselines

### Root cause: matrix jobs overwrite each other (fixed in `llmci` 0.1.1)

Parallel CI matrix jobs used to race on a single PR comment — last job won. **Scaffold 0.1.1+** merges slices when `LLMCI_REPORT_SLICE` is set (testbed workflow already uses `${{ matrix.service }}/${{ matrix.config }}`). Upgrade PyPI: `pip install --upgrade llmci`.

### 2.3 Fix `--config` documentation (or implement CLI flag)

**Problem:** Service READMEs document `llmci run --config scaffold-*.yaml`, but **`llmci` has no `--config` flag**. Only `shared/scripts/scaffold_run.sh --config ...` works.

**Affected files:**

| File | Current (wrong) |
|------|-----------------|
| `services/ticket-classifier/README.md` | `llmci run --config scaffold-prompt.yaml` |
| `services/support-agent/README.md` | `llmci run --config scaffold-single.yaml` |
| `services/summarizer/README.md` | `llmci run --config scaffold-mock.yaml` |

**Option A — Testbed-only (quick):** Replace all with:

```bash
MOCK_LLM=1 ../../shared/scripts/scaffold_run.sh --config scaffold-prompt.yaml
```

**Option B — llmci CLI (better long-term):** Add `--config` to `llmci` in the main Scaffold repo, publish `0.1.1`, then simplify testbed CI/Makefile to call `llmci run --config` directly and delete `scaffold_run.sh`.

**Acceptance:**
- [ ] Every documented run command works copy-paste on a fresh clone
- [ ] CI and README use the same invocation pattern

---

## 3. P1 — Demo and dogfood value

### 3.1 Create demo regression branches

Create from green `main` after §2 is complete. Each branch should fail CI and produce a **Scaffold PR comment** with failed examples.

| Branch | File change | Expected failure |
|--------|-------------|------------------|
| `test/break-classifier` | Remove or gut `billing` keywords in `services/ticket-classifier/app/classifier.py` (`CATEGORY_KEYWORDS`) | `service-classification` accuracy below 0.75 |
| `test/break-rag-retrieval` | Force empty retrieval in `services/rag-qa/pipeline/retrieve.py` (e.g. `top_k=0` or return `[]`) | `rag-qa` `pass_rate` below 0.80 |
| `test/break-agent-safety` | In `services/support-agent/agent/run_agent.py`, call `delete_account` on normal queries | `support-agent-single` constraint / `mean_score` failure |

**Workflow per branch:**

```bash
git checkout main && git pull
git checkout -b test/break-classifier
# make one targeted change
git commit -am "demo: break classifier billing keywords"
git push -u origin test/break-classifier
# open PR → verify Scaffold comment
```

**Acceptance:**
- [ ] Three open PRs (or merged demo PRs) showing red CI + PR comment table
- [ ] README demo table (§ Demo regression branches) matches actual branches

### 3.2 Align `make eval-all` with CI matrix

**Problem:** `Makefile` `eval-all` omits prompt-level ticket classifier. CI runs it.

**File:** `Makefile`

**Add:**

```makefile
eval-classifier-prompt:
	cd services/ticket-classifier && MOCK_LLM=1 ../../shared/scripts/scaffold_run.sh --config scaffold-prompt.yaml
```

Include in `eval-all` before or after `eval-classifier`.

**Acceptance:**
- [ ] `make eval-all` covers the same configs as CI matrix (7 eval runs)

### 3.3 CI badge in README

After GitHub repo exists, add to root `README.md`:

```markdown
[![llmci Evals](https://github.com/<org>/scaffold-testbed/actions/workflows/scaffold.yml/badge.svg)](https://github.com/<org>/scaffold-testbed/actions/workflows/scaffold.yml)
```

---

## 4. P2 — Correctness and spec alignment

### 4.1 Reconcile baselines with current configs

**Problem:** Some committed baselines include metrics not in current YAML (e.g. support-agent baselines have `pass_rate`; configs only define `mean_score`).

**Task:** On `main`, re-run `--update-baseline` for every service/config after configs are final, then commit.

**Acceptance:**
- [ ] Each baseline JSON metrics keys match active `scaffold*.yaml` metric names

### 4.2 Support-agent config vs outline

**Outline §8.4** suggests `mean_score` + `error_rate` for CI configs. Current `scaffold-single.yaml` / `scaffold-multi.yaml` only have `mean_score`.

**Decision:** Add `error_rate` threshold if you want outline parity, or document intentional simplification (constraint-only CI).

### 4.3 Summarizer mock metrics vs outline

**Outline §7.4** lists `mean_score`, `min_score`, `cosine_similarity` (with refs). Current `scaffold-mock.yaml` uses `mean_score` + `pass_rate` only.

**Decision:** Either expand mock judge to expose those metrics, or note in service README that mock config intentionally simplifies.

### 4.4 Support-agent code fixes

| Issue | File | Notes |
|-------|------|-------|
| Trace step numbering | `agent/run_agent.py` | After `_append_tool()` returns incremented step, extra `step += 1` may skip step numbers in trace |
| History injection ignored | `agent/run_agent.py` | `run_multi_turn()` uses `user_message` only; `scaffold-history.yaml` won't work for context-dependent turns ("Can you cancel it?") until `history` is wired in |
| No default config | `services/support-agent/` | Bare `llmci run` fails — document or add symlink/default `llmci.yaml` |

**Acceptance for history mode (optional):**
- [ ] `scaffold-history.yaml` added to manual LLM workflow once history routing works
- [ ] At least one multi-turn scenario requires prior context to pass

### 4.5 Summarizer footgun

`services/summarizer/llmci.yaml` is the **real LLM judge** config. Running `llmci run` without `OPENAI_API_KEY` fails.

**Task:** README should lead with mock config; warn that default `llmci.yaml` requires API key. Consider renaming or adding a comment at top of `llmci.yaml`.

---

## 5. P3 — Nice-to-have polish

### 5.1 Link from Scaffold main repo

Per outline §18, after testbed is public:

| Scaffold file | Change |
|---------------|--------|
| `README.md` | "Reference integration" section → testbed repo |
| `docs.html` | Case study sections → `scaffold-testbed/services/...` links |
| `TESTBED.md` | Point to live testbed URL (not just outline) |

### 5.2 `wait_for_http.sh`

Present in `shared/scripts/` but unused. Wire into:
- `docker-compose.yml` healthcheck or documented HTTP eval flow for ticket-classifier (`CLASSIFIER_URL=http://localhost:8000`), or
- Remove if not planned

### 5.3 Docker-compose vs outline

Current `docker-compose.yml` uses inline `uvicorn` command (no Dockerfile). Outline §15 shows `build: ./services/ticket-classifier`. Only needed if you want containerized HTTP testing in CI.

### 5.4 Extend LLM workflow

`scaffold-llm.yml` currently runs:
- summarizer (real judge)
- migration (`llmci run` only — not `llmci migrate`)
- support-agent (`scaffold-single-full.yaml` only)

Consider adding: multi-turn composite config, `scaffold-history.yaml`, or a separate manual job for `llmci migrate`.

### 5.5 Richer datasets

`json-api` has 5 eval rows; outline implies minimal demo. Expand if you want more impressive PR comment tables.

### 5.6 Secrets for LLM workflow

Ensure `OPENAI_API_KEY` is set as a GitHub repository secret before running `scaffold-llm.yml`.

---

## 6. Scaffold main repo changes (optional, helps testbed)

These are **not** in testbed but reduce friction:

| Change | Benefit |
|--------|---------|
| Add `llmci run --config PATH` to `llmci` | Remove `scaffold_run.sh` workaround |
| Publish `llmci==0.1.1` with `--config` | Testbed CI simplifies |
| Trusted PyPI publishing workflow | Safer releases than manual tokens |

See [PYPI.md](./PYPI.md) for publish process.

---

## 7. Verification checklist (definition of done)

Run on a fresh clone after all P0 + P1 work:

```bash
git clone git@github.com:<org>/scaffold-testbed.git
cd scaffold-testbed
make install
make test                    # 4 passed
make eval-all                # all mock evals green
```

On GitHub:
- [ ] Push to `main` → CI green
- [ ] PR from `test/break-classifier` → CI red + Scaffold PR comment
- [ ] PR from `test/break-rag-retrieval` → CI red + PR comment
- [ ] PR from `test/break-agent-safety` → CI red + PR comment
- [ ] Manual `scaffold-llm.yml` dispatch succeeds with `OPENAI_API_KEY` secret

---

## 8. Suggested implementation order

1. Git init + push + green CI on `main`
2. `fetch-depth: 0` + `--compare-to=origin/main` in CI
3. Refresh baselines on `main` with real commit SHAs
4. Fix `--config` docs (or ship CLI flag)
5. `make eval-all` parity with CI matrix
6. Create three `test/break-*` branches + verify PR comments
7. CI badge + link from Scaffold main repo
8. Agent/summarizer polish (P2/P3) as time allows

---

## 9. File index (quick reference)

```
scaffold-testbed/
├── .github/workflows/scaffold.yml      # P0: compare-to, fetch-depth
├── .github/workflows/scaffold-llm.yml    # P3: optional extensions
├── Makefile                              # P1: eval-all parity
├── README.md                             # P1: CI badge, PyPI install
├── shared/scripts/scaffold_run.sh        # P0: pass --compare-to through
├── services/ticket-classifier/
│   ├── app/classifier.py                 # P1: break-classifier target
│   └── README.md                         # P0: fix --config docs
├── services/rag-qa/pipeline/retrieve.py  # P1: break-rag-retrieval target
├── services/support-agent/
│   ├── agent/run_agent.py                # P1/P2: break-agent-safety, history
│   └── README.md                         # P0: fix --config docs
└── services/summarizer/README.md         # P0/P2: mock vs real config
```
