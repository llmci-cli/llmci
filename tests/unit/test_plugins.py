"""Tests for the judge plugin / extension API."""

import pytest

from llmci.errors import ConfigError
from llmci.judges.base import Judge
from llmci.judges.factory import create_judge
from llmci.models import EvalExample, JudgeConfig, JudgeResult, TargetResult
from llmci.plugins import (
    BUILTIN_JUDGE_TYPES,
    get_judge_factory,
    register_judge,
    registered_judge_types,
    reset_registry,
)


class _KeywordJudge(Judge):
    """Toy plugin judge: 1.0 if the expected keyword is in the output."""

    async def evaluate_single(self, input: str, expected: str, actual: str) -> JudgeResult:
        hit = expected.lower() in actual.lower()
        return JudgeResult(score=1.0 if hit else 0.0)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


class TestRegister:
    def test_register_subclass_and_build(self):
        register_judge("keyword", _KeywordJudge)
        judge = create_judge(JudgeConfig(type="keyword"))
        assert isinstance(judge, _KeywordJudge)

    def test_register_factory_callable_receives_config(self):
        seen = {}

        def factory(config):
            seen["model"] = config.model
            return _KeywordJudge()

        register_judge("kw2", factory)
        create_judge(JudgeConfig(type="kw2", model="gpt-4o"))
        assert seen["model"] == "gpt-4o"

    def test_empty_name_rejected(self):
        with pytest.raises(ConfigError):
            register_judge("", _KeywordJudge)

    def test_builtin_collision_rejected(self):
        for builtin in BUILTIN_JUDGE_TYPES:
            with pytest.raises(ConfigError, match="built-in"):
                register_judge(builtin, _KeywordJudge)

    def test_conflicting_reregistration_rejected(self):
        register_judge("kw", _KeywordJudge)

        class _Other(Judge):
            pass

        with pytest.raises(ConfigError, match="already registered"):
            register_judge("kw", _Other)

    def test_idempotent_same_factory(self):
        def factory(config):
            return _KeywordJudge()

        register_judge("kw", factory)
        register_judge("kw", factory)  # same object, no error
        assert "kw" in registered_judge_types()

    def test_non_callable_rejected(self):
        with pytest.raises(ConfigError):
            register_judge("bad", 123)  # type: ignore[arg-type]


class TestFactoryIntegration:
    def test_unknown_type_lists_builtins_and_plugins(self):
        register_judge("keyword", _KeywordJudge)
        with pytest.raises(ConfigError) as exc:
            create_judge(JudgeConfig(type="does_not_exist"))
        msg = str(exc.value)
        assert "exact_match" in msg
        assert "keyword" in msg  # registered plugin surfaced in the hint

    def test_plugin_build_error_wrapped(self):
        def factory(config):
            raise RuntimeError("boom")

        register_judge("explodes", factory)
        with pytest.raises(ConfigError, match="failed to build"):
            create_judge(JudgeConfig(type="explodes"))

    def test_builtin_still_works(self):
        from llmci.judges.exact_match import ExactMatchJudge

        assert isinstance(create_judge(JudgeConfig(type="exact_match")), ExactMatchJudge)


async def test_plugin_judge_scores_examples():
    register_judge("keyword", _KeywordJudge)
    judge = create_judge(JudgeConfig(type="keyword"))
    examples = [
        EvalExample(input="q", expected="paris"),
        EvalExample(input="q", expected="london"),
    ]
    results = [
        TargetResult(output="The answer is Paris.", latency_ms=1.0),
        TargetResult(output="The answer is Berlin.", latency_ms=1.0),
    ]
    per_example = await judge.evaluate_dataset(examples, results)
    assert per_example[0].score == 1.0
    assert per_example[1].score == 0.0


def test_get_judge_factory_returns_none_for_unknown():
    assert get_judge_factory("nope") is None


class TestConfigPluginLoading:
    def test_load_config_imports_plugin_modules(self, tmp_path, monkeypatch):
        # A local plugin module that registers a judge on import.
        plugin_dir = tmp_path / "plug_pkg"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "judges.py").write_text(
            "from llmci.judges.base import Judge\n"
            "from llmci.models import JudgeResult\n"
            "from llmci.plugins import register_judge\n"
            "class MyJudge(Judge):\n"
            "    async def evaluate_single(self, input, expected, actual):\n"
            "        return JudgeResult(score=1.0)\n"
            "register_judge('my_local', MyJudge)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        config_path = tmp_path / "llmci.yaml"
        config_path.write_text(
            "version: 1\n"
            "plugins: [plug_pkg.judges]\n"
            "target:\n  command: 'echo hi > {output_file}'\n"
            "evals:\n"
            "  - name: e\n    dataset: ./d.jsonl\n    judge: {type: my_local}\n"
        )

        from llmci.config import load_config

        config = load_config(config_path)
        assert "my_local" in registered_judge_types()
        judge = create_judge(config.evals[0].judge)
        assert judge.__class__.__name__ == "MyJudge"
