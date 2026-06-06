# 13: Custom Judge & Metric Plugins

Registers a project-specific `judge.type` **and** a custom metric **without forking
llmci**, using the plugin API. The `json_schema` judge checks that each response is valid
JSON containing the required keys; the `json_field_coverage` metric tracks the mean
fraction of required keys present.

## What it shows
- A local judge plugin registered via `register_judge(...)`
- A local metric plugin registered via `register_metric(...)`, gateable by name
- Loading both with a `plugins:` list in `llmci.yaml` (no packaging needed)

## How it works
`eval_plugins.py` defines a `Judge` subclass and a metric function, then calls
`register_judge("json_schema", ...)` and `register_metric("json_field_coverage", ...)` at
import time. `llmci.yaml` lists the module under `plugins:`, so llmci imports it during
config load and both become available. The dataset's `expected` field lists the required
top-level keys.

A metric function receives a `MetricContext` (examples, target results, judge results,
and the indices/scores of non-errored examples) and returns one aggregate float. Pass
`lower_is_better=True` to `register_metric` to flip the threshold direction (like cost or
latency).

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
