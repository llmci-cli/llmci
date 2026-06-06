# 19 · Cross-provider migration

Migrate a prompt from one **provider** to another (not just a model version bump on the
same API). Uses the same eval dataset and `llmci migrate` optimizer as
[`examples/02-model-migration`](../02-model-migration/).

## Provider/model syntax

Pass litellm-style `provider/model` identifiers:

```bash
cd examples/02-model-migration   # reuses the ticket-classification eval + prompt

export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...

llmci migrate \
  --from openai/gpt-4o-mini \
  --to anthropic/claude-3-haiku-20240307 \
  --eval ticket-classification
```

## Internal proxies

When source and target hit different gateways, set per-side base URLs:

```bash
llmci migrate \
  --from openai/gpt-4o-mini --from-base-url https://openai-proxy.internal/v1 \
  --to anthropic/claude-3-haiku-20240307 --to-base-url https://anthropic-proxy.internal/v1 \
  --eval ticket-classification
```

## Few-shot strategy

Instead of rewriting the prompt, greedily add train examples as inline few-shot demos:

```bash
llmci migrate \
  --from openai/gpt-4o-mini \
  --to anthropic/claude-3-haiku-20240307 \
  --eval ticket-classification \
  --strategy few_shot \
  --max-few-shot 5
```

The optimized `prompt.txt` will include a `## Examples` block above the task instructions.
