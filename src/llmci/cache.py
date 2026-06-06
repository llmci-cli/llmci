"""Content-addressed response cache for direct API targets.

Re-running CI should not re-pay for unchanged examples. This cache keys a stored
model response on the tuple that fully determines it — model, prompt, and any
request parameters that affect output — so identical calls across runs are served
from disk instead of the provider.

Caching applies to *direct* (litellm) targets only. Command-mode targets run
arbitrary scripts that may have side effects, so they are never cached here.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE_DIR = Path(".llmci/cache/responses")

# Bump when the on-disk entry shape changes so stale entries are ignored.
CACHE_VERSION = 1


@dataclass
class CachedResponse:
    """A response replayed from cache."""

    output: str
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0


def make_key(
    *,
    model: str,
    prompt: str,
    base_url: str | None = None,
    params: dict | None = None,
) -> str:
    """Build a stable cache key from the inputs that determine a response.

    The prompt already has the example input substituted in, so it is part of the
    key. ``params`` captures anything else that changes output (temperature, etc.).
    Operational knobs that do not change output (timeout, retries) are excluded on
    purpose so they don't fragment the cache.
    """
    payload = {
        "v": CACHE_VERSION,
        "model": model,
        "prompt": prompt,
        "base_url": base_url,
        "params": params or {},
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """A simple file-backed response cache.

    Each entry is a small JSON file named by its key under ``cache_dir``. The cache
    is intentionally dependency-free and safe to commit-ignore; a corrupt or
    unreadable entry is treated as a miss rather than an error.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        enabled: bool = True,
        refresh: bool = False,
    ) -> None:
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.enabled = enabled
        # ``refresh`` forces a miss on read (so live calls overwrite entries) while
        # still writing fresh results back.
        self.refresh = refresh
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> CachedResponse | None:
        """Return a cached response, or None on miss/disabled/refresh."""
        if not self.enabled or self.refresh:
            return None

        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None

        try:
            data = json.loads(path.read_text())
            entry = CachedResponse(
                output=data["output"],
                latency_ms=float(data.get("latency_ms", 0.0)),
                tokens_in=int(data.get("tokens_in", 0)),
                tokens_out=int(data.get("tokens_out", 0)),
                cost=float(data.get("cost", 0.0)),
            )
        except (OSError, ValueError, KeyError):
            self.misses += 1
            return None

        self.hits += 1
        return entry

    def set(
        self,
        key: str,
        output: str,
        latency_ms: float,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Store a response. No-op when disabled."""
        if not self.enabled:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        entry = {
            "output": output,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": cost,
            "cached_at": time.time(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False))
        tmp.replace(path)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
