# 13: Custom Judge Plugin

Registers a project-specific `judge.type` **without forking llmci**, using the plugin
API. The `json_schema` judge checks that each response is valid JSON containing the
required keys.

## What it shows
- A local judge plugin registered via `register_judge(...)`
- Loading it with a `plugins:` list in `llmci.yaml` (no packaging needed)
- Referencing the plugin as `judge: {type: json_schema}`

## How it works
`eval_plugins.py` defines a `Judge` subclass and calls `register_judge("json_schema", ...)`
at import time. `llmci.yaml` lists the module under `plugins:`, so llmci imports it during
config load and the new judge type becomes available. The dataset's `expected` field lists
the required top-level keys.

## How to run
```bash
cd examples/13-plugin-judge
llmci run
```

## Distributing a plugin
Instead of a `plugins:` list, a published package can advertise the judge via an entry
point — it's then discovered automatically once installed:

```toml
# pyproject.toml of your plugin package
[project.entry-points."llmci.judges"]
json_schema = "my_pkg.judges:JsonSchemaJudge"
```

## How to adapt
- A plugin value can be a `Judge` subclass or a `(JudgeConfig) -> Judge` factory.
- Plugin types can't shadow built-ins (`exact_match`, `llm`, `rag`, `safety`, …).
