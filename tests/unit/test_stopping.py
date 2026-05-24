"""Tests for early stopping."""

from llmci.migrate.stopping import EarlyStopping


class TestEarlyStopping:
    def test_initial_score_accepted(self):
        es = EarlyStopping(patience=3)
        assert es.should_stop(0.5) is False
        assert es.best_score == 0.5

    def test_improvement_resets_patience(self):
        es = EarlyStopping(patience=3, min_improvement=0.01)
        es.should_stop(0.5)
        es.should_stop(0.49)  # stale
        es.should_stop(0.49)  # stale
        assert es.stale_count == 2
        es.should_stop(0.52)  # improvement
        assert es.stale_count == 0

    def test_patience_exhausted(self):
        es = EarlyStopping(patience=2, min_improvement=0.01)
        assert es.should_stop(0.5) is False
        assert es.should_stop(0.5) is False   # stale 1
        assert es.should_stop(0.5) is True    # stale 2 = patience

    def test_min_improvement_threshold(self):
        es = EarlyStopping(patience=3, min_improvement=0.05)
        es.should_stop(0.5)
        assert es.should_stop(0.52) is False  # stale: 0.02 < 0.05
        assert es.stale_count == 1

    def test_monotonic_improvement(self):
        es = EarlyStopping(patience=3, min_improvement=0.01)
        for score in [0.5, 0.55, 0.60, 0.65, 0.70]:
            assert es.should_stop(score) is False
        assert es.best_score == 0.70

    def test_is_improving(self):
        es = EarlyStopping(patience=3)
        es.should_stop(0.5)
        assert es.is_improving is True
        es.should_stop(0.5)
        assert es.is_improving is False
