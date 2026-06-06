# 16 · Structured-output (JSON Schema) judging

Gate on whether your LLM emits **valid structured output**. The built-in `structured` judge
parses the target's JSON answer and validates it against a JSON Schema — scoring 1.0 when
valid and 0.0 otherwise. Fully deterministic and API-key-free.

## Run it

```bash
llmci run --config llmci.yaml
```

`run_extractor.py` is a mock extractor that returns schema-valid product records, so
`accuracy` (mean validity) is **1.0** and the gate passes.

## Watch it fail

Set `BROKEN = True` in `run_extractor.py` and re-run. The extractor emits a record with a
**string** `price` and a **missing** `id`; the schema validator reports both and the score
drops to 0.0:

```
product-extraction.price: expected type number, got string; ... missing required property 'id'
```

## Configuring the judge

The schema lives inline under `json_schema:` (or point it at a `.json` file with
`json_schema: ./schema.json`). The validator supports the practical JSON-Schema subset:
`type` (incl. lists), `required`, `properties`, `additionalProperties`, `items`, `enum`,
`minimum`/`maximum`, `minLength`/`maxLength`, `minItems`/`maxItems`, and `pattern`.

Set `partial_credit: true` on the judge to score the **fraction** of required top-level
fields that validate instead of pass/fail — useful for tracking how close outputs are
while a feature is still being built.
