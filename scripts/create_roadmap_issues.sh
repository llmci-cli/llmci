#!/usr/bin/env bash
#
# Create the "Now" tier roadmap issues (items 1-4) on GitHub.
#
# Prerequisite: an authenticated GitHub CLI.
#   gh auth login -h github.com
#
# Usage:
#   scripts/create_roadmap_issues.sh
#
# Idempotency: this script does NOT dedupe. Run it once. If a run fails partway,
# check `gh issue list --label roadmap` before re-running.

set -euo pipefail

REPO="llmci-cli/llmci"

if ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated. Run: gh auth login -h github.com" >&2
  exit 1
fi

# Ensure labels exist (no-op if already present).
ensure_label() {
  gh label create "$1" --repo "$REPO" --color "$2" --description "$3" 2>/dev/null || true
}
ensure_label "roadmap" "5319e7" "Roadmap item"
ensure_label "now-tier" "b60205" "Now: highest-leverage roadmap work"

create() {
  local title="$1"
  local body="$2"
  gh issue create --repo "$REPO" \
    --title "$title" \
    --label "roadmap,now-tier" \
    --body "$body"
}

create "Flake resistance: multi-sample runs + statistical significance" \
'## Why
LLM outputs are nondeterministic; a single run can fail a threshold by noise, and a
flaky gate gets turned off. This is the #1 existential risk for a pre-merge gate.

## Scope
- `settings.samples_per_example` to run each example N times.
- Aggregate with mean + variance; surface per-metric confidence intervals.
- `max_regression` compares against baseline using a significance test, not a raw
  point delta. Add a `significance: 0.95` knob.
- Report shows "drop not statistically significant" instead of a hard fail on noise.

## Status
A working prototype has landed: `settings.samples_per_example` / `--samples` run each
eval over multiple rounds; metrics report a mean + confidence interval
(`src/llmci/significance.py`); `settings.significance` / `--significance` gates
`max_regression` so drops within noise are reported but not enforced. Remaining
follow-ups:
- Store baseline sample distributions for a true two-sample test (current gate uses a
  one-sample CI against a point baseline).
- Per-example bootstrap as an alternative to the round-level normal-approx CI.
- Extend sampling to agent evals.

## Acceptance
- Re-running an identical PR does not flip pass/fail due to sampling noise.
- Reports display CI bounds for sampled metrics.

Roadmap: ROADMAP.md → Now → item 1. Effort: M. Impact: ★★★.'

create "Response caching for direct API targets" \
'## Why
Re-running CI should not re-pay for unchanged examples. Caching cuts PR cost and
wall-clock and makes multi-sample / pairwise evaluation affordable.

## Scope
- Cache keyed on `(provider, model, prompt, input, params)`.
- Store under `.llmci/cache/responses/`; `--no-cache` and `--refresh-cache` flags.
- Applies to direct (litellm) targets only; command-mode targets may have side effects.

## Status
A working prototype has landed: `src/llmci/cache.py`, wired through the runner and the
`run` command (`--no-cache`, `--refresh-cache`). Remaining follow-ups:
- Cache eviction / max-age policy.
- Include sampling/temperature params in the key once multi-sample lands.
- Document cache behavior in README.

Roadmap: ROADMAP.md → Now → item 2. Effort: S. Impact: ★★★.'

create "Cost / token budgeting as a first-class metric" \
'## Why
A pre-merge gate should catch cost regressions, not just quality ones.

## Scope
- New metrics: `cost_total`, `cost_mean`, `tokens_in_mean`, `tokens_out_mean`.
- Thresholds in both `absolute` and `max_regression` modes
  (e.g. "cost-per-eval may not rise >20% vs baseline").
- Pull token/price data from litellm; allow a price override table for proxies.

## Status
A working prototype has landed: `cost_total`, `cost_mean`, `tokens_in_mean`,
`tokens_out_mean`, `tokens_total_mean` metrics; cost/usage captured from litellm
responses (direct) or optional `usage`/`cost` output keys (command); and a
lower-is-better direction concept so `absolute`/`max_regression` gate cost correctly.
Remaining follow-ups:
- Price override table for internal proxies / unknown models.
- Per-example cost breakdown in the report.

## Acceptance
- `llmci run` reports per-eval cost and token counts.
- A PR that raises cost beyond the threshold fails the gate.

Roadmap: ROADMAP.md → Now → item 3. Effort: M. Impact: ★★★.'

create "CI portability: JUnit XML + SARIF output" \
'## Why
Everything currently assumes GitHub Actions + PR comments. Structured output unlocks
every CI system (GitLab, Bitbucket, Azure DevOps, Jenkins, CircleCI) for free.

## Scope
- `llmci run --output-format markdown|junit|sarif|json`.
- JUnit XML → native test reporting on all major CI providers.
- SARIF → code-scanning surfaces for regressions tied to changed files.

## Status
A working prototype has landed: `src/llmci/report_formats.py` with junit/sarif/json
formatters, wired through `run --output-format` (PR comments stay markdown). Remaining
follow-ups:
- Map SARIF results to changed files/locations for inline annotations.
- Document recipes for GitLab/Bitbucket/Azure DevOps in the README.

Roadmap: ROADMAP.md → Now → item 4. Effort: S. Impact: ★★★.'

echo "Done. View: gh issue list --repo $REPO --label roadmap"
