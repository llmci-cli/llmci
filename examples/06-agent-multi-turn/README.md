# 06: Agent Multi-Turn

Demonstrates evaluation of a conversational agent across multiple turns.

## What it shows
- Multi-turn agent evaluation with `full_replay` mode
- Per-turn expected outcomes and constraints
- Conversation-level constraint budgets
- Cumulative history passed to each turn

## How to run
```bash
cd examples/06-agent-multi-turn
scaffold run
```

## How to adapt
- Replace `run_agent.py` with your conversational agent
- Agent receives `{"user_message": "...", "history": [...], "turn_index": N}`
- Define conversations in `evals/conversations.jsonl` with `turns` array
- Choose `full_replay` (command called per turn) or `history_injection` (command called once)
