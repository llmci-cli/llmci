"""Shared, content-addressed cache for LLM judge calls.

Re-running CI shouldn't re-pay for judge LLM calls on unchanged outputs. The ``llm`` judge
has long cached its own results; this provides the same benefit to the other LLM-based
judges (pairwise, RAG, safety) through one helper so they cache consistently.

The cache reuses :class:`llmci.cache.ResponseCache` (keyed on model + prompt) but writes to
a separate ``.llmci/cache/judges`` directory, and honors the same ``--no-cache`` /
``--refresh-cache`` flags as target caching via :func:`judge_cache_from`.
"""

from __future__ import annotations

import litellm

from llmci.cache import DEFAULT_CACHE_DIR, ResponseCache, make_key

JUDGE_CACHE_DIR = DEFAULT_CACHE_DIR.parent / "judges"


def judge_cache_from(target_cache: ResponseCache | None) -> ResponseCache | None:
    """Build a judge-call cache that mirrors a target cache's enabled/refresh flags.

    Returns None when target caching is disabled/absent, so judges fall back to live calls
    (and existing call-counting tests that pass no cache are unaffected).
    """
    if target_cache is None or not target_cache.enabled:
        return None
    return ResponseCache(
        JUDGE_CACHE_DIR, enabled=True, refresh=target_cache.refresh
    )


async def complete(
    model: str,
    prompt: str,
    *,
    cache: ResponseCache | None = None,
    temperature: float = 0.0,
    timeout: int = 30,
) -> str:
    """Return the judge model's completion text, served from cache when available.

    Errors propagate to the caller (judges wrap this in their own try/except and never
    cache a failed call).
    """
    if cache is not None:
        key = make_key(model=model, prompt=prompt, params={"temperature": temperature})
        hit = cache.get(key)
        if hit is not None:
            return hit.output

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=timeout,
    )
    content = response.choices[0].message.content or ""

    if cache is not None:
        cache.set(key, content, 0.0)
    return content
