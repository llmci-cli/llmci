"""Statistical helpers for multi-sample eval runs.

LLM outputs are nondeterministic, so a single run can pass or fail a threshold by
chance. When an eval is run for several rounds, each round yields one metric value;
this module turns that sample into a confidence interval so the gate can tell a real
regression from run-to-run noise.

The CI uses a normal approximation (mean ± z · sd/√n), which is deterministic and
dependency-free — important for reproducible CI gates and stable tests.
"""

from __future__ import annotations

import statistics
from statistics import NormalDist


def mean(values: list[float]) -> float:
    """Arithmetic mean; 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a ``(low, high)`` confidence interval for the mean of ``values``.

    With fewer than two values the interval collapses to the point estimate, since
    there is no spread to estimate. ``confidence`` is the two-sided level, e.g. 0.95.
    """
    if not values:
        return (0.0, 0.0)

    m = mean(values)
    n = len(values)
    if n < 2:
        return (m, m)

    sd = statistics.stdev(values)
    if sd == 0.0:
        return (m, m)

    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    margin = z * sd / (n ** 0.5)
    return (m - margin, m + margin)
