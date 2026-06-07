"""Direct API target runner via litellm."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import litellm

from llmci.cache import ResponseCache, make_key
from llmci.models import EvalExample, TargetResult
from llmci.multimodal import build_user_content, has_media, media_cache_params
from llmci.pricing import resolve_cost


async def run_direct_target(
    provider: str,
    model: str,
    prompt_template: str,
    examples: list[EvalExample],
    parallelism: int = 10,
    timeout: int = 30,
    retries: int = 2,
    base_url: str | None = None,
    cache: ResponseCache | None = None,
    media_base: Path | None = None,
    price_overrides: dict[str, dict[str, float]] | None = None,
) -> list[TargetResult]:
    """Run a direct API target on all examples with bounded concurrency.

    When ``cache`` is provided, identical (model, prompt, base_url) calls are
    served from disk instead of hitting the provider.
    """
    semaphore = asyncio.Semaphore(parallelism)
    model_str = f"{provider}/{model}" if provider else model

    async def run_one(example: EvalExample) -> TargetResult:
        async with semaphore:
            return await _run_single_direct(
                model_str, prompt_template, example, timeout, retries,
                base_url=base_url, cache=cache, media_base=media_base,
                price_overrides=price_overrides,
            )

    return await asyncio.gather(*[run_one(ex) for ex in examples])


async def _run_single_direct(
    model_str: str,
    prompt_template: str,
    example: EvalExample,
    timeout: int,
    retries: int,
    base_url: str | None = None,
    cache: ResponseCache | None = None,
    media_base: Path | None = None,
    price_overrides: dict[str, dict[str, float]] | None = None,
) -> TargetResult:
    """Run a single example through litellm with retries, using the cache if set."""
    last_error: str | None = None

    prompt = prompt_template.replace("{input}", example.input)
    try:
        content = build_user_content(prompt, example, media_base)
    except ValueError as e:
        return TargetResult(output="", latency_ms=0.0, error=str(e))

    cache_params = media_cache_params(example) if has_media(example) else None
    cache_key = (
        make_key(
            model=model_str,
            prompt=prompt,
            base_url=base_url,
            params=cache_params,
        )
        if cache is not None
        else None
    )

    if cache is not None and cache_key is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return TargetResult(
                output=hit.output,
                latency_ms=hit.latency_ms,
                tokens_in=hit.tokens_in,
                tokens_out=hit.tokens_out,
                cost=hit.cost,
            )

    for attempt in range(retries + 1):
        try:
            start = time.monotonic()

            kwargs: dict = {
                "model": model_str,
                "messages": [{"role": "user", "content": content}],
                "timeout": timeout,
            }
            if base_url:
                kwargs["api_base"] = base_url

            response = await litellm.acompletion(**kwargs)

            elapsed_ms = (time.monotonic() - start) * 1000
            content = (response.choices[0].message.content or "").strip()
            tokens_in, tokens_out = _extract_usage(response)
            cost = resolve_cost(
                model=model_str,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                litellm_cost=_extract_cost(response),
                price_overrides=price_overrides,
            )

            if cache is not None and cache_key is not None:
                cache.set(
                    cache_key, content, elapsed_ms,
                    tokens_in=tokens_in, tokens_out=tokens_out, cost=cost,
                )

            return TargetResult(
                output=content,
                latency_ms=elapsed_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
            )

        except Exception as e:
            last_error = f"LLM API error: {e}"
            if attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))

    return TargetResult(output="", latency_ms=0.0, error=last_error)


def _extract_usage(response: object) -> tuple[int, int]:
    """Pull (prompt_tokens, completion_tokens) from a litellm response.

    Returns (0, 0) when usage is unavailable.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return (0, 0)
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    try:
        return (int(prompt), int(completion))
    except (TypeError, ValueError):
        return (0, 0)


def _extract_cost(response: object) -> float:
    """Pull the response cost from litellm, falling back to 0.0.

    litellm attaches a computed cost under ``_hidden_params['response_cost']`` for
    providers it knows pricing for; otherwise we leave cost at 0.0.
    """
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        cost = hidden.get("response_cost")
        if cost is not None:
            try:
                return float(cost)
            except (TypeError, ValueError):
                return 0.0
    return 0.0
