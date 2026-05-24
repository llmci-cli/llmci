"""Subprocess-based target runner (black box mode)."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

from llmci.errors import TargetError
from llmci.models import EvalExample, TargetResult


async def run_command_target(
    command_template: str,
    examples: list[EvalExample],
    parallelism: int = 10,
    timeout: int = 30,
    retries: int = 2,
) -> list[TargetResult]:
    """Run a command-mode target on all examples with bounded concurrency."""
    semaphore = asyncio.Semaphore(parallelism)

    async def run_one(example: EvalExample) -> TargetResult:
        async with semaphore:
            return await _run_single_command(command_template, example, timeout, retries)

    return await asyncio.gather(*[run_one(ex) for ex in examples])


async def _run_single_command(
    command_template: str,
    example: EvalExample,
    timeout: int,
    retries: int,
) -> TargetResult:
    """Run a single example through the command target with retries."""
    last_error: str | None = None

    for attempt in range(retries + 1):
        try:
            return await _execute_command(command_template, example, timeout)
        except (TargetError, asyncio.TimeoutError) as e:
            last_error = str(e)
            if attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))

    return TargetResult(output="", latency_ms=0.0, error=last_error)


async def _execute_command(
    command_template: str,
    example: EvalExample,
    timeout: int,
) -> TargetResult:
    """Execute a single command subprocess."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as input_f, tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as output_f:
        input_path = Path(input_f.name)
        output_path = Path(output_f.name)

    try:
        input_data = {"input": example.input, "expected": example.expected, **example.extra}
        input_path.write_text(json.dumps(input_data))
        output_path.write_text("")

        cmd = command_template.replace("{input_file}", str(input_path)).replace(
            "{output_file}", str(output_path)
        )

        start = time.monotonic()

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TargetError(f"Command timed out after {timeout}s: {cmd}")

        elapsed_ms = (time.monotonic() - start) * 1000

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()
            raise TargetError(
                f"Command exited with code {proc.returncode}: {cmd}\n"
                f"stderr: {stderr_text[:500]}"
            )

        raw_output = output_path.read_text().strip()
        if not raw_output:
            raise TargetError(f"Command produced empty output file: {output_path}")

        try:
            output_data = json.loads(raw_output)
            output_text = output_data.get("output", raw_output)
        except json.JSONDecodeError:
            output_text = raw_output

        return TargetResult(output=str(output_text), latency_ms=elapsed_ms)

    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
