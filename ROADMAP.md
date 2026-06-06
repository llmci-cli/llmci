# llmci Roadmap

This roadmap turns llmci's core thesis — **evals as a pre-merge safety gate, not an
observability dashboard** — into a prioritized plan. Items are ordered by leverage:
how much each one strengthens the "this belongs in CI" story relative to effort.

The strategic bet: win decisively on the *gate experience* (trustworthy, cheap,
portable, fast) rather than chase feature-surface parity with promptfoo / DeepEval.

Legend: **Effort** S (days) · M (1–2 weeks) · L (multi-week). **Impact** ★–★★★.

---

## Now — make the gate trustworthy and portable

These four close the gaps most likely to make a team *disable* the gate, or block
adoption outside GitHub. Highest leverage, mostly low/medium effort.

> **Status:** all four Now-tier items have a working prototype landed (see CHANGELOG
> `[Unreleased]`). Remaining follow-ups are tracked in `scripts/create_roadmap_issues.sh`.

### 1. Flake resistance: multi-sample runs + statistical significance — `M` · ★★★ ✅ prototype
LLM outputs are nondeterministic; a single run can fail a threshold by noise, and a
flaky gate gets turned off. This is the #1 existential risk.
- `settings.samples_per_example` to run each example N times.
- Aggregate with mean + variance; surface per-metric confidence intervals.
- `max_regression` compares against baseline using a significance test (e.g. bootstrap
  CI overlap), not a raw point delta. Add `significance: 0.95` knob.
- Report shows "drop not statistically significant" instead of a hard fail on noise.

### 2. Response caching — `S` · ★★★ ✅ prototype
Re-running CI should not re-pay for unchanged examples.
- Cache keyed on `(provider, model, prompt, input, params)`.
- Store under `.llmci/cache/responses/`; `--no-cache` and `--refresh-cache` flags.
- Cuts PR cost and wall-clock dramatically; also makes multi-sample (item 1) affordable.

### 3. Cost / token budgeting as a first-class metric — `M` · ★★★ ✅ prototype
A pre-merge gate should catch cost regressions, not just quality ones.
- New metrics: `cost_total`, `cost_mean`, `tokens_in_mean`, `tokens_out_mean`.
- Thresholds in both `absolute` and `max_regression` modes (e.g. "cost-per-eval may
  not rise >20% vs baseline").
- Pull token/price data from litellm; allow a price override table for proxies.

### 4. CI portability: JUnit XML + SARIF output — `S` · ★★★ ✅ prototype
Everything currently assumes GitHub Actions + PR comments. Structured output unlocks
*every* CI system (GitLab, Bitbucket, Azure DevOps, Jenkins, CircleCI) for free.
- `llmci run --output-format junit|sarif|json`.
- JUnit XML → native test reporting on all major CI providers.
- SARIF → code-scanning surfaces for regressions tied to changed files.

---

## Next — deepen eval quality and reviewer experience

Once the gate is trustworthy, raise the ceiling on *what* it can catch and how
reviewers act on failures.

> **Status:** all five Next-tier items (5–9) have a working prototype landed (see
> CHANGELOG `[Unreleased]`). Remaining follow-ups are noted per item below.

### 5. RAG-specific judges — `L` · ★★★ ✅ prototype
The biggest demand gap vs DeepEval / Ragas.
- Faithfulness / groundedness, context relevance, answer relevance, retrieval
  recall@k / precision@k.
- Ship as built-in judge types so RAG pipelines (already testable via `command` mode)
  get first-class metrics.

_Landed: `judge: {type: rag}` with the five criteria above; each surfaces as a
gateable metric by name. Targets pass `contexts`/`retrieved_ids`; gold labels via
`relevant_ids`. Follow-ups: a runnable `examples/` case, judge-result caching for the
LLM criteria, and per-claim faithfulness decomposition._

### 6. Output diff view — baseline vs PR, per example — `M` · ★★ ✅ prototype
"Why did this fail?" is currently hard to answer from a pass/fail table.
- Example-level diff of baseline output vs current output for regressed examples.
- Surface in the PR comment (collapsible) and in a standalone report (item 9).

_Landed: baselines store per-example outputs; markdown + HTML reports show an
"Output Diffs vs Baseline" section (matched by input, worst regressions first).
Backward compatible with older baselines._

### 7. Pairwise / preference judging — `M` · ★★ ✅ prototype
Absolute scoring is weak for open-ended generation.
- `judge: {type: pairwise}` to evaluate "is B better than A" vs baseline output.
- Win-rate metric with significance reuse from item 1.

_Landed: compares current vs baseline output per input; `win_rate` surfaced as a
gateable metric; reuses per-example baseline outputs (item 6) and CI sampling (item 1).
Follow-up: order-swap to control position bias, and judge-result caching._

### 8. LLM-judge calibration & drift detection — `M` · ★★ ✅ prototype
LLM judges drift across model versions and disagree with humans; trust erodes silently.
- `llmci judge calibrate` to measure judge↔human agreement on a labeled set.
- Detect judge-score drift when the judge model changes; warn in CI.

_Landed: `llmci judge calibrate --eval <name> --labels <file>` runs the eval's judge
over a human-labeled set and reports agreement rate, Cohen's kappa, MAE, and Pearson r.
A snapshot (`.llmci/calibration/<eval>.json`, via `--save-snapshot`) records the judge
model + scores; a later run flags drift when the model changes. Gate with
`--min-agreement` / `--max-drift`. Follow-up: per-criterion calibration for composite/RAG
judges, and trend history across snapshots._

### 9. Shareable HTML/Markdown run report — `S` · ★★ ✅ prototype
A self-contained artifact beyond the PR comment.
- `llmci run --output-format html --output report.html`; upload as a CI artifact.
- Includes the summary table (with CIs), regressions, and per-example results.

_Landed as an `--output-format html` value (inline CSS, no external assets). Follow-up:
fold in the item-6 baseline-vs-current per-example diff once that lands._

---

## Later — broaden surface and safety

Valuable, but only after the gate is sticky. Several overlap with where promptfoo /
DeepEval are already strong, so they're differentiators-of-degree, not of-kind.

### 10. Safety / red-team assertions — `L` · ★★
- PII leakage, jailbreak resistance, toxicity checks as judge types / assertions.
- Optional adversarial input generation for a `llmci redteam` mode.

### 11. Multimodal & structured-output evals — `M` · ★
- Image/audio inputs through litellm; JSON-schema / structured-output judging built in
  (today this requires a custom judge).

### 12. More migration targets & strategies — `M` · ★★
- Migration across providers (not just model versions), and few-shot/example-selection
  optimization in addition to prompt rewriting.

### 13. Plugin / extension API — `M` · ★
- Stable entry-point API for third-party judges, metrics, and report sinks so the
  ecosystem can extend llmci without forking.

---

## Cross-cutting / continuous

- **Docs & examples** keep pace with each feature (the testbed monorepo is the proving
  ground — every new judge/metric should land there first).
- **Determinism guardrails**: warn when a config has no baseline, or thresholds with no
  significance config once item 1 lands.
- **Performance**: keep PR feedback under a few minutes; caching (item 2) and parallelism
  are the levers.

---

## Sequencing rationale

1. Items **1–4** are the moat. "The LLM eval tool that actually belongs in CI" requires
   it to be flake-resistant, cheap, cost-aware, and CI-agnostic. Ship these first.
2. Caching (2) is a prerequisite that makes multi-sample (1) and pairwise (7) affordable.
3. RAG judges (5) is the largest demand pull but is L-effort, so it follows the quick
   trust/portability wins.
4. Safety/red-team (10) and multimodal (11) are deferred: real demand, but they compete
   on someone else's turf and shouldn't precede solidifying the core differentiator.
