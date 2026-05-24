# 05: Agent Single-Turn

Demonstrates evaluation of a tool-using agent with constraint-based composite judging.

## What it shows
- `level: agent` evaluation
- Composite judge with constraint checking
- Tool call budget validation (max_tool_calls, required_tools, forbidden_tools)
- Agent trace format (tool calls + responses)

## How to run
```bash
cd examples/05-agent-single-turn
llmci run
```

## How to adapt
- Replace `run_agent.py` with your agent
- Agent must output JSON with `final_output`, `trace`, `total_tool_calls`, `total_tokens`
- Define constraints in `evals/scenarios.jsonl`
- Add outcome or trajectory criteria to the composite judge for LLM-based evaluation
