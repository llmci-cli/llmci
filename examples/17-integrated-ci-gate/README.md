# 17 · Integrated CI gate

One realistic pre-merge gate that stacks the **Now-tier** features together — not as
isolated demos, but as the kind of `llmci.yaml` you'd run on every PR.

| Concern | Eval | How it's gated |
|---------|------|----------------|
| **Quality** | `support-routing` | `accuracy` ≥ 75% (`exact_match`) |
| **Cost regression** | `support-routing` | `cost_mean` may not rise >25% vs baseline |
| **Token regression** | `support-routing` | `tokens_in_mean` may not rise >25% vs baseline |
| **Safety** | `support-safety` | `pii_leakage` = 1.0 (deterministic PII scan) |

Fully deterministic and **API-key-free**. Committed baselines under `.llmci/baselines/`
make `max_regression` thresholds meaningful out of the box.

## Run the gate

```bash
# Uses the committed baselines in .llmci/baselines/ automatically
llmci run --config llmci.yaml

# On a PR branch, compare against main's baselines instead:
llmci run --config llmci.yaml --compare-to=origin/main

# Refresh baselines after an intentional improvement lands on main
llmci run --config llmci.yaml --update-baseline
```

## Portable CI output

The same run can feed any CI system:

```bash
llmci run --config llmci.yaml --compare-to=HEAD \
  --output-format junit --output results.xml
llmci run --config llmci.yaml --compare-to=HEAD \
  --output-format html --output report.html
```

## Watch each gate fail

**Quality** — break classification in `run_classifier.py` (return `"general"` always) and
`accuracy` drops below 0.75.

**Cost regression** — set `INFLATE_COST = True` in `run_classifier.py`. Token usage triples;
`cost_mean` and `tokens_in_mean` exceed the 25% `max_regression` allowance vs baseline.

**Safety** — set `LEAK_PII = True` in `run_responder.py`. The reply dumps fake PII and
`pii_leakage` falls to 0.0.

## What this doesn't demo (by design)

**Response caching** and **flake resistance** (`samples_per_example` + `significance`)
apply to *direct API* targets where LLM outputs vary between runs. This example uses
command-mode mocks so it runs without API keys; see
`tests/integration/test_runner_caching_flake.py` for end-to-end coverage of caching and
multi-sample CI through the runner (with a mocked provider).

On a real service, you'd add those `settings:` knobs to this same config:

```yaml
settings:
  samples_per_example: 3
  significance: 0.95
```
