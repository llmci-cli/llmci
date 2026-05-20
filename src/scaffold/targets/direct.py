"""Direct API target runner via litellm."""

from __future__ import annotations

import asyncio
import time

import litellm

from scaffold.models import EvalExample, TargetResult


async def run_direct_target(
    provider: str,
    model: str,
    prompt_template: str,
    examples: list[EvalExample],
    parallelism: int = 10,
    timeout: int = 30,
    retries: int = 2,
) -> list[TargetResult]:
    """Run a direct API target on all examples with bounded concurrency."""
    semaphore = asyncio.Semaphore(parallelism)
    model_str = f"{provider}/{model}" if provider else model

    async def run_one(example: EvalExample) -> TargetResult:
        async with semaphore:
            return await _run_single_direct(
                model_str, prompt_template, example, timeout, retries
            )

    return await asyncio.gather(*[run_one(ex) for ex in examples])


async def _run_single_direct(
    model_str: str,
    prompt_template: str,
    example: EvalExample,
    timeout: int,
    retries: int,
) -> TargetResult:
    """Run a single example through litellm with retries."""
    last_error: str | None = None

    for attempt in range(retries + 1):
        try:
            prompt = prompt_template.replace("{input}", example.input)

            start = time.monotonic()

            response = await litellm.acompletion(
                model=model_str,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            content = response.choices[0].message.content or ""

            return TargetResult(output=content.strip(), latency_ms=elapsed_ms)

        except Exception as e:
            last_error = f"LLM API error: {e}"
            if attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))

    return TargetResult(output="", latency_ms=0.0, error=last_error)
