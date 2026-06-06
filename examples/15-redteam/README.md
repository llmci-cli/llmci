# 15 · Red-team attack generation

Active safety testing: instead of hand-writing adversarial prompts, **generate** them from
a few plain seed intents, then gate the responses with the `safety` judge. Fully
deterministic and API-key-free.

## 1. Generate the attack dataset

```bash
llmci redteam generate \
  --seeds seeds.txt \
  --category pii_extraction --category injection \
  --output evals/attacks.jsonl
```

Each seed in `seeds.txt` is expanded into many adversarial framings (jailbreak,
prompt-injection, PII-extraction, obfuscation). The committed `evals/attacks.jsonl` is the
output of the command above — regenerate it any time and diff the result in CI.

See everything available with:

```bash
llmci redteam generate --list
```

### Optional: LLM-mutated variants

For broader coverage beyond the built-in library, append rephrased variants (requires an
API key; cached under `.llmci/cache/judges/` by default):

```bash
llmci redteam generate \
  --seeds seeds.txt \
  --category pii_extraction \
  --mutate --mutate-count 2 \
  --output evals/attacks-mutated.jsonl
```

Mutated rows include `parent_attack` and `mutated: true` so failures trace back to the
technique they were derived from.

## 2. Gate the responses

```bash
llmci run --config llmci.yaml
```

`run_assistant.py` is a mock assistant that refuses to reveal personal data under every
framing, so the deterministic `pii_leakage` criterion stays at **1.0** and the gate passes.

## 3. Watch it fail

Set `LEAKY = True` in `run_assistant.py` and re-run: the assistant "complies" with the
extraction attacks and dumps a record full of PII, so `pii_leakage` drops to **0.0** and
the gate fails — exactly what you want a safety regression to do on a PR.

## How it fits together

`llmci redteam generate` produces the adversarial **inputs**; the `safety` judge scores the
**outputs**. Generated rows carry `attack`, `category`, and `seed` metadata so a failure can
be traced back to the specific technique that broke through. Add LLM-based criteria
(`toxicity`, `jailbreak_resistance`) to the judge to cover harmful-content and refusal
quality beyond deterministic PII scanning.
