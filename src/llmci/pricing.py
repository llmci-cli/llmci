"""Cost estimation helpers for direct (litellm) targets."""

from __future__ import annotations

from typing import Mapping


def resolve_cost(
    *,
    model: str,
    tokens_in: int,
    tokens_out: int,
    litellm_cost: float,
    price_overrides: Mapping[str, Mapping[str, float]] | None,
) -> float:
    """Return per-call cost, preferring litellm pricing then configured overrides."""
    if litellm_cost > 0:
        return litellm_cost
    if not price_overrides:
        return 0.0

    rates = _lookup_rates(model, price_overrides)
    if rates is None:
        return 0.0

    input_rate = rates.get("input_per_token")
    output_rate = rates.get("output_per_token")
    if input_rate is None and output_rate is None:
        return 0.0

    total = 0.0
    if input_rate is not None:
        total += tokens_in * float(input_rate)
    if output_rate is not None:
        total += tokens_out * float(output_rate)
    return round(total, 8)


def _lookup_rates(
    model: str,
    price_overrides: Mapping[str, Mapping[str, float]],
) -> Mapping[str, float] | None:
    if model in price_overrides:
        return price_overrides[model]
    if "/" in model:
        short = model.split("/", 1)[1]
        if short in price_overrides:
            return price_overrides[short]
    return None
