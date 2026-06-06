"""Tests for migration model spec parsing."""

from llmci.migrate.model_spec import ModelSpec


class TestModelSpec:
    def test_bare_model(self):
        spec = ModelSpec.parse("gpt-4o-mini")
        assert spec.provider == ""
        assert spec.model == "gpt-4o-mini"
        assert spec.raw == "gpt-4o-mini"
        assert spec.base_url is None

    def test_provider_model(self):
        spec = ModelSpec.parse("anthropic/claude-3-haiku-20240307")
        assert spec.provider == "anthropic"
        assert spec.model == "claude-3-haiku-20240307"
        assert spec.raw == "anthropic/claude-3-haiku-20240307"

    def test_base_url(self):
        spec = ModelSpec.parse("openai/gpt-4o", "https://proxy.internal/v1")
        assert spec.base_url == "https://proxy.internal/v1"
