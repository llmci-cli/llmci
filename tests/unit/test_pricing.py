"""Tests for direct-target cost estimation with price overrides."""

from llmci.pricing import resolve_cost


def test_prefers_litellm_cost_when_available():
    cost = resolve_cost(
        model="openai/gpt-4o-mini",
        tokens_in=1000,
        tokens_out=200,
        litellm_cost=0.05,
        price_overrides={"gpt-4o-mini": {"input_per_token": 1.0, "output_per_token": 1.0}},
    )
    assert cost == 0.05


def test_uses_override_when_litellm_cost_missing():
    cost = resolve_cost(
        model="openai/gpt-4o-mini",
        tokens_in=1000,
        tokens_out=200,
        litellm_cost=0.0,
        price_overrides={
            "openai/gpt-4o-mini": {
                "input_per_token": 0.000001,
                "output_per_token": 0.000002,
            }
        },
    )
    assert cost == 0.0014


def test_short_model_key_fallback():
    cost = resolve_cost(
        model="openai/gpt-4o-mini",
        tokens_in=500,
        tokens_out=0,
        litellm_cost=0.0,
        price_overrides={"gpt-4o-mini": {"input_per_token": 0.000002}},
    )
    assert cost == 0.001


def test_returns_zero_without_rates():
    cost = resolve_cost(
        model="unknown/model",
        tokens_in=100,
        tokens_out=50,
        litellm_cost=0.0,
        price_overrides={"gpt-4o-mini": {"input_per_token": 0.001}},
    )
    assert cost == 0.0
