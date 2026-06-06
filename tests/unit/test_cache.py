"""Tests for the response cache."""

from llmci.cache import ResponseCache, make_key


class TestMakeKey:
    def test_stable_for_same_inputs(self):
        a = make_key(model="openai/gpt-4o", prompt="hello")
        b = make_key(model="openai/gpt-4o", prompt="hello")
        assert a == b

    def test_differs_by_prompt(self):
        a = make_key(model="openai/gpt-4o", prompt="hello")
        b = make_key(model="openai/gpt-4o", prompt="world")
        assert a != b

    def test_differs_by_model(self):
        a = make_key(model="openai/gpt-4o", prompt="hello")
        b = make_key(model="openai/gpt-4o-mini", prompt="hello")
        assert a != b

    def test_differs_by_base_url(self):
        a = make_key(model="m", prompt="p", base_url=None)
        b = make_key(model="m", prompt="p", base_url="https://proxy/v1")
        assert a != b


class TestResponseCache:
    def test_miss_then_hit(self, tmp_path):
        cache = ResponseCache(cache_dir=tmp_path)
        key = make_key(model="m", prompt="p")

        assert cache.get(key) is None
        assert cache.misses == 1

        cache.set(key, "the answer", 123.0)
        hit = cache.get(key)

        assert hit is not None
        assert hit.output == "the answer"
        assert hit.latency_ms == 123.0
        assert cache.hits == 1

    def test_disabled_never_reads_or_writes(self, tmp_path):
        cache = ResponseCache(cache_dir=tmp_path, enabled=False)
        key = make_key(model="m", prompt="p")

        cache.set(key, "x", 1.0)
        assert cache.get(key) is None
        assert not list(tmp_path.glob("*.json"))

    def test_refresh_forces_miss_but_still_writes(self, tmp_path):
        seed = ResponseCache(cache_dir=tmp_path)
        key = make_key(model="m", prompt="p")
        seed.set(key, "old", 1.0)

        cache = ResponseCache(cache_dir=tmp_path, refresh=True)
        assert cache.get(key) is None

        cache.set(key, "new", 2.0)
        replay = ResponseCache(cache_dir=tmp_path)
        hit = replay.get(key)
        assert hit is not None
        assert hit.output == "new"

    def test_corrupt_entry_is_a_miss(self, tmp_path):
        cache = ResponseCache(cache_dir=tmp_path)
        key = make_key(model="m", prompt="p")
        (tmp_path / f"{key}.json").write_text("not json{")

        assert cache.get(key) is None
