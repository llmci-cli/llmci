"""Tests for the red-team attack generator."""

import base64

import pytest

from llmci.errors import ConfigError
from llmci.redteam import (
    BUILTIN_ATTACKS,
    attack_categories,
    attack_names,
    generate_attacks,
    load_seeds,
    write_attacks,
)


class TestLibrary:
    def test_every_template_has_distinct_name(self):
        names = [a.name for a in BUILTIN_ATTACKS]
        assert len(names) == len(set(names))

    def test_categories_and_names_sorted(self):
        assert attack_categories() == sorted(attack_categories())
        assert attack_names() == sorted(attack_names())

    def test_templates_embed_seed(self):
        for tmpl in BUILTIN_ATTACKS:
            rendered = tmpl.render("PAYLOAD")
            # Obfuscation templates transform the seed, so they won't contain it verbatim.
            if tmpl.category != "obfuscation":
                assert "PAYLOAD" in rendered

    def test_base64_template_is_decodable(self):
        tmpl = next(a for a in BUILTIN_ATTACKS if a.name == "base64_wrap")
        rendered = tmpl.render("do the thing")
        token = rendered.rsplit(" ", 1)[-1]
        assert base64.b64decode(token).decode() == "do the thing"


class TestGenerate:
    def test_expands_every_template_per_seed(self):
        rows = generate_attacks(["a", "b"])
        assert len(rows) == 2 * len(BUILTIN_ATTACKS)
        assert {r["seed"] for r in rows} == {"a", "b"}
        assert all(r["attack"] in attack_names() for r in rows)

    def test_category_filter(self):
        rows = generate_attacks(["x"], categories=["injection"])
        assert rows
        assert {r["category"] for r in rows} == {"injection"}

    def test_attack_filter(self):
        rows = generate_attacks(["x"], attacks=["roleplay_dan"])
        assert [r["attack"] for r in rows] == ["roleplay_dan"]

    def test_include_control_prepends_raw_seed(self):
        rows = generate_attacks(["x"], attacks=["roleplay_dan"], include_control=True)
        assert rows[0] == {"input": "x", "attack": "none", "category": "control", "seed": "x"}
        assert len(rows) == 2

    def test_blank_seeds_skipped(self):
        rows = generate_attacks(["", "  ", "real"], attacks=["roleplay_dan"])
        assert [r["seed"] for r in rows] == ["real"]

    def test_unknown_category_rejected(self):
        with pytest.raises(ConfigError, match="Unknown attack categories"):
            generate_attacks(["x"], categories=["nope"])

    def test_unknown_attack_rejected(self):
        with pytest.raises(ConfigError, match="Unknown attacks"):
            generate_attacks(["x"], attacks=["nope"])

    def test_no_match_raises(self):
        with pytest.raises(ConfigError, match="No attack templates"):
            generate_attacks(["x"], categories=["injection"], attacks=["roleplay_dan"])


class TestSeedsAndWrite:
    def test_load_text_seeds_ignores_comments_and_blanks(self, tmp_path):
        p = tmp_path / "seeds.txt"
        p.write_text("# comment\n\nfirst\nsecond\n")
        assert load_seeds(p) == ["first", "second"]

    def test_load_jsonl_seeds_accepts_aliases(self, tmp_path):
        p = tmp_path / "seeds.jsonl"
        p.write_text('{"input": "a"}\n{"seed": "b"}\n{"prompt": "c"}\n')
        assert load_seeds(p) == ["a", "b", "c"]

    def test_load_jsonl_missing_field_raises(self, tmp_path):
        p = tmp_path / "seeds.jsonl"
        p.write_text('{"foo": "a"}\n')
        with pytest.raises(ConfigError, match="needs an"):
            load_seeds(p)

    def test_write_attacks_roundtrips(self, tmp_path):
        rows = generate_attacks(["x"], attacks=["roleplay_dan"])
        out = tmp_path / "nested" / "attacks.jsonl"
        n = write_attacks(rows, out)
        assert n == 1
        loaded = load_seeds(out)  # rows have an "input" field
        assert loaded == [rows[0]["input"]]
