# 12: RAG Retrieval Metrics

Gates a retrieval-augmented pipeline on **retrieval quality** using the built-in `rag`
judge. The `retrieval_recall` and `retrieval_precision` criteria are deterministic — no
API key required.

## What it shows
- The `rag` judge type with retrieval criteria scored `@k`
- `retrieval_recall` / `retrieval_precision` surfaced as gateable metrics by name
- A command target emitting structured RAG output (`output`, `contexts`, `retrieved_ids`)
- Gold retrieval labels supplied per dataset row as `relevant_ids`

## How it works
`run_rag.py` retrieves passages from a tiny fixed corpus by keyword overlap and writes:

```json
{"output": "<answer>", "contexts": ["..."], "retrieved_ids": ["doc1", "doc2"]}
```

Each dataset row carries the gold set:

```json
{"input": "What is the capital of France?", "relevant_ids": ["doc1"]}
```

The judge compares `retrieved_ids` against `relevant_ids` to compute recall@k and
precision@k. No model is called for these criteria.

## How to run
```bash
cd examples/12-rag-retrieval
llmci run
```

## How to adapt
- Add `faithfulness` / `answer_relevance` / `context_relevance` criteria and set a
  `model:` to also score the generated answer (these call the judge model).
- Tune `k`, or weight criteria with `weight:` to shape the overall score.
