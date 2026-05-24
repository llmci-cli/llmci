# 04: Custom Judge

Demonstrates evaluation with a user-defined Python judge function for domain-specific validation.

## What it shows
- Custom judge loaded from a Python file
- JSON schema validation as a judge criterion
- Programmatic scoring logic

## How to run
```bash
cd examples/04-custom-judge
llmci run
```

## How to adapt
- Write your own judge function in a `.py` file
- Function signature: `def judge(input: str, expected: str, actual: str) -> dict`
- Return `{"score": 0.0-1.0, "reason": "..."}`
- Point `llmci.yaml` at your module and function
