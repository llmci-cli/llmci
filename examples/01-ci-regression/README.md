# 01: CI Regression Testing

Demonstrates Scaffold's core use case — a deterministic classifier evaluated with exact_match judging and F1/accuracy metrics.

## What it shows
- Command-mode target (runs a Python script)
- `exact_match` judge
- `accuracy` and `f1_macro` metrics with absolute thresholds

## How to run
```bash
cd examples/01-ci-regression
scaffold run
```

## How to adapt
- Replace `run_prompt.py` with your own classifier/LLM script
- Edit `evals/tickets.jsonl` with your evaluation examples
- Adjust thresholds in `scaffold.yaml`
