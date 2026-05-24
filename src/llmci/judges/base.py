"""Base judge interface and result types."""

from __future__ import annotations

from llmci.models import EvalExample, JudgeResult, TargetResult


class Judge:
    """Base interface for all judges. All methods are async for uniformity."""

    async def evaluate_single(
        self, input: str, expected: str, actual: str
    ) -> JudgeResult:
        """Score a single example."""
        raise NotImplementedError

    async def evaluate_dataset(
        self,
        examples: list[EvalExample],
        results: list[TargetResult],
    ) -> list[JudgeResult]:
        """Score all examples, skipping target errors.

        Default: calls evaluate_single on each valid pair.
        Subclasses can override for batch evaluation.
        """
        per_example: list[JudgeResult] = []
        for ex, res in zip(examples, results):
            if res.error is not None:
                per_example.append(JudgeResult(score=0.0, reason=f"Target error: {res.error}"))
            else:
                result = await self.evaluate_single(ex.input, ex.expected, res.output)
                per_example.append(result)
        return per_example
