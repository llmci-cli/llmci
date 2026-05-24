"""Agent target runners for single-turn and multi-turn evaluation."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

from llmci.errors import TargetError
from llmci.models import AgentScenario, AgentTrace, TraceStep


async def run_agent_target(
    command_template: str,
    scenarios: list[AgentScenario],
    mode: str = "full_replay",
    parallelism: int = 5,
    timeout: int = 60,
) -> list[AgentTrace]:
    """Run agent target on all scenarios with bounded concurrency."""
    semaphore = asyncio.Semaphore(parallelism)

    async def run_one(scenario: AgentScenario) -> AgentTrace:
        async with semaphore:
            if scenario.is_multi_turn:
                return await _run_multi_turn(
                    command_template, scenario, mode, timeout
                )
            return await _run_single_turn(command_template, scenario, timeout)

    return list(await asyncio.gather(*[run_one(s) for s in scenarios]))


async def _run_single_turn(
    command_template: str,
    scenario: AgentScenario,
    timeout: int,
) -> AgentTrace:
    """Execute a single-turn agent scenario."""
    input_data = (
        scenario.input
        if isinstance(scenario.input, dict)
        else {"input": scenario.input}
    )

    try:
        raw = await _execute_agent_command(command_template, input_data, timeout)
        return _parse_trace(raw)
    except (TargetError, Exception) as e:
        return AgentTrace(error=str(e))


async def _run_multi_turn(
    command_template: str,
    scenario: AgentScenario,
    mode: str,
    timeout: int,
) -> AgentTrace:
    """Execute a multi-turn agent scenario."""
    if not scenario.turns:
        return AgentTrace(error="No turns defined")

    if mode == "history_injection":
        return await _run_history_injection(command_template, scenario, timeout)
    return await _run_full_replay(command_template, scenario, timeout)


async def _run_full_replay(
    command_template: str,
    scenario: AgentScenario,
    timeout: int,
) -> AgentTrace:
    """Full replay: invoke the command once per turn with cumulative history."""
    if not scenario.turns:
        return AgentTrace(error="No turns defined")

    history: list[dict] = []
    per_turn: list[dict] = []
    all_steps: list[TraceStep] = []
    total_tool_calls = 0
    total_tokens = 0
    total_latency = 0.0

    for i, turn in enumerate(scenario.turns):
        input_data = {
            "user_message": turn.user_message,
            "history": history,
            "turn_index": i,
        }
        if turn.context:
            input_data["context"] = turn.context

        try:
            raw = await _execute_agent_command(command_template, input_data, timeout)
            turn_trace = _parse_trace(raw)

            history.append({"role": "user", "content": turn.user_message})
            if turn_trace.final_output:
                history.append({"role": "assistant", "content": turn_trace.final_output})

            per_turn.append({
                "turn": i,
                "final_output": turn_trace.final_output,
                "trace": [s.model_dump() for s in turn_trace.trace],
                "tool_calls": turn_trace.total_tool_calls,
                "tokens": turn_trace.total_tokens,
            })

            all_steps.extend(turn_trace.trace)
            total_tool_calls += turn_trace.total_tool_calls
            total_tokens += turn_trace.total_tokens
            total_latency += turn_trace.latency_ms

        except (TargetError, Exception) as e:
            per_turn.append({"turn": i, "error": str(e)})

    last_output = per_turn[-1].get("final_output") if per_turn else None

    return AgentTrace(
        final_output=last_output,
        trace=all_steps,
        total_tool_calls=total_tool_calls,
        total_tokens=total_tokens,
        latency_ms=total_latency,
        turns=per_turn,
    )


async def _run_history_injection(
    command_template: str,
    scenario: AgentScenario,
    timeout: int,
) -> AgentTrace:
    """History injection: send all turns at once, only the final turn is executed."""
    if not scenario.turns:
        return AgentTrace(error="No turns defined")

    pre_history = []
    for turn in scenario.turns[:-1]:
        pre_history.append({"role": "user", "content": turn.user_message})
        pre_history.append({"role": "assistant", "content": "(prior response)"})

    final_turn = scenario.turns[-1]
    input_data = {
        "user_message": final_turn.user_message,
        "history": pre_history,
        "turn_index": len(scenario.turns) - 1,
    }
    if final_turn.context:
        input_data["context"] = final_turn.context

    try:
        raw = await _execute_agent_command(command_template, input_data, timeout)
        trace = _parse_trace(raw)
        trace.turns = [{
            "turn": len(scenario.turns) - 1,
            "final_output": trace.final_output,
            "trace": [s.model_dump() for s in trace.trace],
            "tool_calls": trace.total_tool_calls,
            "tokens": trace.total_tokens,
        }]
        return trace
    except (TargetError, Exception) as e:
        return AgentTrace(error=str(e))


async def _execute_agent_command(
    command_template: str,
    input_data: dict,
    timeout: int,
) -> dict:
    """Execute the agent command subprocess and return parsed output."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as input_f, tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as output_f:
        input_path = Path(input_f.name)
        output_path = Path(output_f.name)

    try:
        input_path.write_text(json.dumps(input_data))
        output_path.write_text("")

        cmd = (
            command_template
            .replace("{input_file}", str(input_path))
            .replace("{output_file}", str(output_path))
        )

        start = time.monotonic()

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TargetError(f"Agent command timed out after {timeout}s")

        elapsed_ms = (time.monotonic() - start) * 1000

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()
            raise TargetError(
                f"Agent command exited with code {proc.returncode}\n"
                f"stderr: {stderr_text[:500]}"
            )

        raw_output = output_path.read_text().strip()
        if not raw_output:
            raise TargetError("Agent command produced empty output file")

        try:
            result: dict = json.loads(raw_output)
            result["_latency_ms"] = elapsed_ms
            return result
        except json.JSONDecodeError as e:
            raise TargetError(f"Agent output is not valid JSON: {e}") from e

    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def _parse_trace(raw: dict) -> AgentTrace:
    """Parse agent command output into an AgentTrace."""
    steps = []
    for i, s in enumerate(raw.get("trace", [])):
        steps.append(TraceStep(
            step=s.get("step", i),
            type=s.get("type", "response"),
            tool=s.get("tool"),
            args=s.get("args"),
            content=s.get("content"),
            tokens=s.get("tokens"),
        ))

    return AgentTrace(
        final_output=raw.get("final_output") or raw.get("output"),
        trace=steps,
        total_tool_calls=raw.get("total_tool_calls", sum(
            1 for s in steps if s.type == "tool_call"
        )),
        total_tokens=raw.get("total_tokens", sum(
            s.tokens or 0 for s in steps
        )),
        latency_ms=raw.get("_latency_ms", 0.0),
    )
