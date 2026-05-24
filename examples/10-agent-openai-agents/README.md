# OpenAI Agents SDK + TraceBuilder

Shows how to run agent evals with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) and convert runs into Scaffold's trace format using `TraceBuilder` and `scaffold.integrations.openai_agents`.

## Quick start (mock — no API key)

```bash
cd examples/10-agent-openai-agents
MOCK_LLM=1 scaffold run
```

Mock mode uses `TraceBuilder` directly with the same tool names as the real agent, so CI and local dev work without `OPENAI_API_KEY`.

## Real OpenAI Agents SDK

```bash
pip install 'llmci[agents]'
export OPENAI_API_KEY=sk-...
cd examples/10-agent-openai-agents
scaffold run
```

## How it fits together

| Layer | Role |
|-------|------|
| **`scaffold.trace.TraceBuilder`** | Framework-neutral way to build valid Scaffold output JSON |
| **`scaffold.integrations.openai_agents`** | Maps SDK `RunResult` → Scaffold output; runs agent from Scaffold input |
| **`run_agent.py`** | Command target Scaffold invokes (`MOCK_LLM=1` or real SDK) |

### Building output manually

```python
from scaffold.trace import TraceBuilder

tb = TraceBuilder()
tb.tool("get_weather", {"city": "London"}, result="58°F cloudy", tokens=25)
tb.response("It's 58°F and cloudy in London.")
output = tb.to_dict()  # write to {output_file}
```

### Using the OpenAI Agents adapter

```python
from agent_def import build_agent
from scaffold.integrations.openai_agents import run_for_scaffold_sync

result = run_for_scaffold_sync(build_agent(), {"query": "Weather in Tokyo?"})
# result has final_output, trace, total_tool_calls, total_tokens
```

## Files

| File | Purpose |
|------|---------|
| `run_agent.py` | Scaffold command target |
| `mock_agent.py` | Mock path with `TraceBuilder` |
| `agent_def.py` | Real OpenAI Agents SDK agent + tools |
| `evals/scenarios.jsonl` | Same constraint scenarios as example 05 |

Maps to docs case study **Agent Testing** and complements `examples/05-agent-single-turn` (hand-rolled mock).
