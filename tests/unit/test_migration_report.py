"""Tests for migration report formatting."""

from llmci.migrate.optimizer import OptimizationResult, OptimizationStep
from llmci.migrate.report import format_migration_report


class TestMigrationReport:
    def test_basic_report(self):
        result = OptimizationResult(
            best_prompt="Improved prompt",
            best_val_score=0.92,
            holdout_score=0.90,
            original_score=0.91,
            from_model="gpt-4o",
            to_model="gpt-4.5",
            steps=[
                OptimizationStep(
                    iteration=1,
                    prompt_text="v1 prompt",
                    train_score=0.85,
                    val_score=0.84,
                    diff="- old\n+ new",
                ),
                OptimizationStep(
                    iteration=2,
                    prompt_text="Improved prompt",
                    train_score=0.90,
                    val_score=0.92,
                    diff="- v1\n+ v2",
                ),
            ],
            stopped_reason="patience",
        )

        report = format_migration_report(result)
        assert "gpt-4o" in report
        assert "gpt-4.5" in report
        assert "patience" in report
        assert "Iteration History" in report
        assert "0.850" in report
        assert "0.900" in report

    def test_parity_achieved(self):
        result = OptimizationResult(
            best_prompt="prompt",
            best_val_score=0.95,
            holdout_score=0.95,
            original_score=0.95,
            from_model="a",
            to_model="b",
            steps=[],
            stopped_reason="converged",
        )
        report = format_migration_report(result)
        assert "Parity achieved" in report

    def test_gap_remaining(self):
        result = OptimizationResult(
            best_prompt="prompt",
            best_val_score=0.80,
            holdout_score=0.75,
            original_score=0.95,
            from_model="a",
            to_model="b",
            steps=[],
            stopped_reason="max_iterations",
        )
        report = format_migration_report(result)
        assert "Gap remaining" in report

    def test_no_steps(self):
        result = OptimizationResult(
            best_prompt="same",
            best_val_score=0.0,
            holdout_score=0.90,
            original_score=0.90,
            from_model="a",
            to_model="b",
            steps=[],
            stopped_reason="converged",
        )
        report = format_migration_report(result)
        assert "Iteration History" not in report

    def test_few_shot_strategy_in_report(self):
        result = OptimizationResult(
            best_prompt="with examples",
            best_val_score=0.9,
            holdout_score=0.88,
            original_score=0.9,
            from_model="openai/gpt-4o",
            to_model="anthropic/claude-3-haiku-20240307",
            strategy="few_shot",
            few_shot_count=3,
        )
        report = format_migration_report(result)
        assert "few-shot" in report
        assert "3 example" in report
