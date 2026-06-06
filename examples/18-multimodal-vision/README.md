# 18 · Multimodal vision eval

Direct API targets can attach **images** (and **audio**) from dataset rows. Paths are
relative to the dataset file; local files are inlined for litellm automatically.

## Dataset format

```json
{"input": "What color?", "expected": "red", "images": ["fixtures/dot.png"]}
```

Paths are relative to the **dataset file's directory** (`evals/` here). `audio` accepts
the same pattern (`.wav`, `.mp3`, `.m4a`, …) for speech models.

## Run (requires API key)

```bash
cd examples/18-multimodal-vision
export OPENAI_API_KEY=...
llmci run
```

The fixture is a 1×1 red PNG. A vision-capable model should answer `red` and pass the
`exact_match` gate.

## How it works

`images` / `audio` fields land in `EvalExample.extra`. The direct target builds a
multimodal litellm message and includes media paths in the response-cache key so
re-runs stay cheap.
