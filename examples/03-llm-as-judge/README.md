# 03: LLM-as-Judge

Demonstrates evaluation of open-ended generation tasks using an LLM judge with rubric-based criteria.

## What it shows
- LLM judge with multi-criteria rubric
- Open-ended evaluation (no single correct answer)
- `mean_score` metric

## How to run
```bash
cd examples/03-llm-as-judge
export OPENAI_API_KEY=your-key
llmci run
```

## How to adapt
- Replace `run_summarizer.py` with your generation pipeline
- Edit the rubric criteria in `llmci.yaml` to match your quality standards
- Add examples to `evals/summaries.jsonl`
