# Example 09: Summarization Quality Assurance

Mirrors the **Summarization QA** case study from the docs.

A content platform generates article summaries using an extractive summarizer.
Quality is evaluated across three dimensions using LLM-as-Judge:

- **Faithfulness** — no hallucinated facts
- **Completeness** — key points covered
- **Conciseness** — no filler or redundancy

## Two evaluation modes

| Eval | Dataset | Description |
|------|---------|-------------|
| `summary-with-refs` | `articles_with_refs.jsonl` | Has human-written reference summaries. Judge compares output to both input and reference. |
| `summary-no-refs` | `articles_no_refs.jsonl` | No reference summaries. Judge evaluates output against the source article only. |

Reference-free evaluation is useful when gold summaries are expensive to create
or when there's no single "correct" summary.

## Run

```bash
# Requires an OpenAI API key for the LLM judge
export OPENAI_API_KEY=sk-...
llmci run
```

## Metrics

This example demonstrates several metrics working together:

- `mean_score` — average quality across all criteria
- `min_score` — catches worst-case summaries (a single terrible summary fails the eval)
- `cosine_similarity` — lightweight token overlap check between generated and reference summaries
- `pass_rate` — fraction of summaries meeting the quality bar

## What to try

1. **Degrade the summarizer**: reduce `MAX_SENTENCES = 1` in `run_summarizer.py` — completeness should drop
2. **Remove lead bias**: set `LEAD_BIAS = 0.0` — see how summary quality changes for news articles
3. **Compare modes**: run both evals and note how reference-free scores differ from reference-based
