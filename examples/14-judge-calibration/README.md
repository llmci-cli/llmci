# 14: Judge Calibration

Measures how well a judge agrees with human labels — and detects drift when the judge
model changes. This example uses the deterministic `exact_match` judge, so it runs with
**no API key**.

## What it shows
- `llmci judge calibrate` over a human-labeled set
- Agreement metrics: agreement rate, Cohen's kappa, MAE, Pearson r
- A calibration snapshot for later drift detection

## How to run
First, the regular gate (a keyword ticket classifier):

```bash
cd examples/14-judge-calibration
llmci run
```

Then calibrate the eval's judge against `labels.jsonl`:

```bash
llmci judge calibrate --eval ticket-classification --labels labels.jsonl --save-snapshot
```

`labels.jsonl` carries the output that was scored plus a human verdict:

```json
{"input": "...", "expected": "account", "output": "security", "human_score": 1}
```

Here the `exact_match` judge marks `output != expected` as a failure, but a human
accepted the answer — so judge and human **disagree** on that row. The report shows an
agreement rate of ~0.83 and moderate Cohen's kappa.

## Drift detection
`--save-snapshot` records the judge model and its scores under
`.llmci/calibration/ticket-classification.json`. Re-running after changing the judge
model reports the mean score change so a model swap can't silently move the gate. Gate it
in CI with `--min-agreement` and/or `--max-drift`:

```bash
llmci judge calibrate --eval ticket-classification --labels labels.jsonl \
  --min-agreement 0.80 --max-drift 0.10
```
