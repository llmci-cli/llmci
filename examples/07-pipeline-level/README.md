# 07: Pipeline-Level Testing

Demonstrates testing a full RAG (Retrieval-Augmented Generation) pipeline end-to-end.

## What it shows
- Pipeline-level evaluation (tests the entire system, not just the prompt)
- Custom judge for flexible answer matching
- Command-mode target wrapping a multi-step pipeline (retrieval + generation)

## How to run
```bash
cd examples/07-pipeline-level
llmci run
```

## How to adapt
- Replace `run_pipeline.py` with your actual pipeline
- The custom judge in `pipeline_judge.py` checks for keyword overlap — adapt the scoring logic to your domain
- Add more QA pairs to `evals/qa.jsonl`
