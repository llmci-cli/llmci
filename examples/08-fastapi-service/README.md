# Example 08: FastAPI Classification Service

Mirrors the **FastAPI Classification Service** case study from the docs.

A customer support ticket classifier with a realistic pipeline:
- **Pre-processing**: text normalization, PII redaction (phone, email, card, SSN)
- **Classification**: keyword-based scoring (simulates an LLM call)
- **Post-processing**: confidence thresholds, fallback routing to "general"

## Why two eval levels?

| Level | Config | What it tests | Speed |
|-------|--------|--------------|-------|
| Prompt | `scaffold-prompt-level.yaml` | Classification logic only | Fast |
| Service | `scaffold.yaml` | Full pipeline (pre/post processing) | Slower |

A prompt-level eval catches prompt regressions in seconds. A service-level eval
catches bugs in PII redaction, confidence thresholds, or other pipeline code.

## Run

```bash
# Service-level (default) — full pipeline
scaffold run

# Prompt-level — swap the config and re-run
cp scaffold-prompt-level.yaml scaffold.yaml
scaffold run
```

## What to try

1. **Break the prompt**: edit `prompts/classify.txt` and re-run the prompt-level eval
2. **Break pre-processing**: change a PII regex in `service.py` and re-run the service-level eval
3. **Lower the confidence threshold**: set `CONFIDENCE_THRESHOLD = 0` in `service.py` — accuracy should change
4. **Compare metrics**: note how `f1_weighted` handles class imbalance differently than `f1_macro`
