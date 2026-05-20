# 02: Model Migration

Demonstrates automated prompt re-tuning when switching between LLM models.

## What it shows
- Direct API mode with a prompt file
- `scaffold migrate` command
- Stratified dataset splitting (train/validation/holdout)
- Iterative prompt optimization with early stopping

## How to run
```bash
cd examples/02-model-migration
export OPENAI_API_KEY=your-key
scaffold migrate --from gpt-4o --to gpt-4.5 --eval ticket-classification
```

## How to adapt
- Edit `prompt.txt` with your production prompt
- Change `--from` and `--to` to your source/target models
- Adjust `--patience` and `--max-iterations` for your use case
