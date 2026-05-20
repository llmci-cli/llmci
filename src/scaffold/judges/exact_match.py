"""Exact match judge for deterministic evaluation."""

from __future__ import annotations

from scaffold.judges.base import Judge
from scaffold.models import JudgeResult


class ExactMatchJudge(Judge):
    """Score is 1.0 if output matches expected (after stripping), else 0.0."""

    def __init__(self, case_sensitive: bool = True):
        self.case_sensitive = case_sensitive

    async def evaluate_single(
        self, input: str, expected: str, actual: str
    ) -> JudgeResult:
        expected_clean = expected.strip()
        actual_clean = actual.strip()

        if not self.case_sensitive:
            expected_clean = expected_clean.lower()
            actual_clean = actual_clean.lower()

        match = expected_clean == actual_clean

        return JudgeResult(
            score=1.0 if match else 0.0,
            reason=None if match else f"Expected '{expected.strip()}', got '{actual.strip()}'",
        )
