"""Early stopping for the optimization loop."""

from __future__ import annotations


class EarlyStopping:
    """Tracks validation scores and determines when to stop."""

    def __init__(self, patience: int = 3, min_improvement: float = 0.005):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_score: float | None = None
        self.stale_count = 0

    def should_stop(self, val_score: float) -> bool:
        """Returns True if optimization should stop."""
        if (
            self.best_score is None
            or val_score > self.best_score + self.min_improvement
        ):
            self.best_score = val_score
            self.stale_count = 0
            return False
        self.stale_count += 1
        return self.stale_count >= self.patience

    @property
    def is_improving(self) -> bool:
        return self.stale_count == 0
