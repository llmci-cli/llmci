# Scaffold Testbed — Next Steps

Updated handoff for [`scaffold-testbed`](https://github.com/alexminnaar/scaffold-testbed) as of **`llmci` 0.1.2** (TraceBuilder + merged PR comment slices).

| Doc | Purpose |
|-----|---------|
| [TESTBED_OUTLINE.md](./TESTBED_OUTLINE.md) | Original build spec |
| [TESTBED_REMAINING.md](./TESTBED_REMAINING.md) | Earlier remaining-work list (partially superseded) |
| **This file** | Current status + what’s actually left |

---

## Status summary

The testbed is **complete** for P0–P2. CI matrix, merged PR comments, and demo PRs are verified.

---

## Completed ✅

| Item | Notes |
|------|-------|
| GitHub repo | [github.com/alexminnaar/scaffold-testbed](https://github.com/alexminnaar/scaffold-testbed) |
| All 5 services + `migration/` | Mock evals pass locally |
| CI matrix (7 eval configs) | `.github/workflows/scaffold.yml` |
| Manual LLM workflow | `.github/workflows/scaffold-llm.yml` |
| Baselines on `main` | Real commit SHAs in `.llmci/baselines/` |
| Regression CI | `--compare-to=origin/main` on PRs; `--update-baseline` on `main` push |
| `fetch-depth: 0` | On committed `main` (needed for git baselines) |
| PyPI dependency | `llmci>=0.1.0`; CI runs `pip install --upgrade llmci` |
| Demo branches | `test/break-classifier`, `test/break-rag-retrieval`, `test/break-agent-safety` |
| CI badge | README |
| `make eval-all` | Matches CI matrix (incl. prompt-level classifier) |
| Service READMEs | Use `scaffold_run.sh`, not `llmci run --config` |
| Scaffold README link | Reference integration section → testbed |

---

## P0 — Fix CI workflow ✅

**Done** on `main` (commit `26349046872` area): `fetch-depth: 0`, `--compare-to=origin/main`, and `LLMCI_REPORT_SLICE` coexist.

### Acceptance

- [x] Change committed and pushed to `main`
- [x] CI installs `llmci` 0.1.2+ (`pip install --upgrade llmci`)
- [x] Demo PRs re-run after fix
- [x] PR comment shows **7 slices** (one per matrix job)
- [x] `test/break-classifier` comment includes ticket-classifier ❌ with failed examples

---

## P1 — Pin minimum `llmci` version ✅

**File:** `scaffold-testbed/pyproject.toml` — `"llmci>=0.1.2"`

---

## P2 — Verify demo branches end-to-end ✅

Verified on re-run (2026-05-24):

| Branch | PR | CI | Merged comment |
|--------|-----|-----|----------------|
| `test/break-classifier` | [#1](https://github.com/alexminnaar/scaffold-testbed/pull/1) | ❌ (expected) | ticket-classifier prompt + service accuracy ❌ |
| `test/break-rag-retrieval` | [#2](https://github.com/alexminnaar/scaffold-testbed/pull/2) | ❌ (expected) | rag-qa pass_rate ❌ |
| `test/break-agent-safety` | [#3](https://github.com/alexminnaar/scaffold-testbed/pull/3) | ❌ (expected) | support-agent constraint ❌ |

---

## P3 — Scaffold main repo (not testbed)

| File | Change |
|------|--------|
| `docs.html` | Link case studies → `scaffold-testbed/services/...` |
| `TESTBED_REMAINING.md` | Mark completed items or archive |

Scaffold README already links to the testbed.

---

## P4 — Optional polish (testbed)

| Item | File / area | Notes |
|------|-------------|-------|
| `OPENAI_API_KEY` secret | GitHub repo settings | Required for `scaffold-llm.yml` |
| LLM workflow slices | `scaffold-llm.yml` | Only if posting PR comments from manual runs |
| Multi-turn LLM eval | `scaffold-llm.yml` | Add `scaffold-multi.yaml` or `scaffold-single-full.yaml` dispatch |
| `llmci migrate` job | New workflow input | Separate from `llmci run` in `migration/` |
| HTTP classifier in CI | `wait_for_http.sh` + docker-compose | Outline §15 stretch |
| Drop `scaffold_run.sh` | All services | When `llmci` adds `llmci run --config` |
| Richer `json-api` dataset | `evals/api_responses.jsonl` | 5 rows → more if desired |
| Pin prod dep | `llmci==0.1.1` | Stricter than `>=`; only if you want frozen customer sim |

---

## Definition of done

```bash
git clone https://github.com/alexminnaar/scaffold-testbed.git
cd scaffold-testbed
make install && make test && make eval-all   # all green
```

On GitHub:

- [ ] `main` CI green
- [ ] Demo PR `#1` (break-classifier): red checks + **merged** PR comment with classifier failures visible
- [ ] Demo PRs for rag + agent: same pattern
- [ ] `llmci` 0.1.1+ in CI logs

---

## Suggested order

1. **P0** — combined workflow fix → push `main`
2. **P1** — bump `llmci>=0.1.1` in testbed `pyproject.toml`
3. **P2** — re-run / verify demo PRs
4. **P3** — `docs.html` links (Scaffold repo)
5. **P4** — as needed

---

## Quick file index

```
scaffold-testbed/
├── .github/workflows/scaffold.yml     ← P0 (critical)
├── pyproject.toml                     ← P1 (llmci>=0.1.1)
├── .github/workflows/scaffold-llm.yml ← P4 (secrets, optional extensions)
└── README.md                          ← P2 (optional demo screenshots)
```
