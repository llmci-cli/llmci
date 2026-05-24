"""Stratified dataset splitting for migration optimization."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from llmci.models import EvalExample


@dataclass
class DataSplit:
    train: list[EvalExample]
    validation: list[EvalExample]
    holdout: list[EvalExample]


def split_dataset(
    examples: list[EvalExample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    """Deterministic stratified split maintaining category distribution."""
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    by_category: dict[str, list[EvalExample]] = defaultdict(list)
    for ex in examples:
        by_category[ex.expected.strip()].append(ex)

    rng = random.Random(seed)
    train: list[EvalExample] = []
    validation: list[EvalExample] = []
    holdout: list[EvalExample] = []

    for _cat, cat_examples in sorted(by_category.items()):
        shuffled = list(cat_examples)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))

        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = max(1, n - n_train - 1)

        train.extend(shuffled[:n_train])
        validation.extend(shuffled[n_train : n_train + n_val])
        holdout.extend(shuffled[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(holdout)

    return DataSplit(train=train, validation=validation, holdout=holdout)
