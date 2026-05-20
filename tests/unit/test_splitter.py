"""Tests for stratified dataset splitting."""

from scaffold.migrate.splitter import split_dataset
from scaffold.models import EvalExample


def _examples(counts: dict[str, int]) -> list[EvalExample]:
    """Create examples with given category counts."""
    examples = []
    for cat, n in counts.items():
        for i in range(n):
            examples.append(EvalExample(input=f"{cat}_{i}", expected=cat))
    return examples


class TestSplitDataset:
    def test_total_preserved(self):
        examples = _examples({"a": 20, "b": 20})
        split = split_dataset(examples)
        total = len(split.train) + len(split.validation) + len(split.holdout)
        assert total == 40

    def test_approximate_ratios(self):
        examples = _examples({"a": 50, "b": 50})
        split = split_dataset(examples)
        assert 60 <= len(split.train) <= 80
        assert 10 <= len(split.validation) <= 25
        assert 10 <= len(split.holdout) <= 25

    def test_stratified(self):
        examples = _examples({"a": 30, "b": 30})
        split = split_dataset(examples)
        train_cats = {ex.expected for ex in split.train}
        val_cats = {ex.expected for ex in split.validation}
        holdout_cats = {ex.expected for ex in split.holdout}
        assert "a" in train_cats and "b" in train_cats
        assert "a" in val_cats and "b" in val_cats
        assert "a" in holdout_cats and "b" in holdout_cats

    def test_deterministic(self):
        examples = _examples({"a": 20, "b": 20})
        s1 = split_dataset(examples, seed=42)
        s2 = split_dataset(examples, seed=42)
        assert [e.input for e in s1.train] == [e.input for e in s2.train]

    def test_different_seeds(self):
        examples = _examples({"a": 20, "b": 20})
        s1 = split_dataset(examples, seed=1)
        s2 = split_dataset(examples, seed=2)
        assert [e.input for e in s1.train] != [e.input for e in s2.train]

    def test_small_dataset(self):
        examples = _examples({"a": 3, "b": 3})
        split = split_dataset(examples)
        assert len(split.train) >= 2
        assert len(split.validation) >= 1
        assert len(split.holdout) >= 1
