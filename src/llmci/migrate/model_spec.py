"""Model identifiers for migration (provider/model + optional proxy base URL)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A litellm model reference with optional per-provider ``base_url``."""

    raw: str
    provider: str
    model: str
    base_url: str | None = None

    @classmethod
    def parse(cls, model: str, base_url: str | None = None) -> ModelSpec:
        """Parse ``provider/model`` or a bare model name."""
        model = model.strip()
        if "/" in model:
            provider, _, name = model.partition("/")
            return cls(raw=model, provider=provider, model=name, base_url=base_url)
        return cls(raw=model, provider="", model=model, base_url=base_url)
