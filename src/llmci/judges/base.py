"""Base judge interface and result types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmci.models import EvalExample, JudgeResult, TargetResult

if TYPE_CHECKING:
    from llmci.cache import ResponseCache


class Judge:
    """Base interface for all judges. All methods are async for uniformity."""

    # Optional shared cache for LLM-based judges; set by the runner. None = no caching.
    _judge_cache: "ResponseCache | None" = None

    def set_judge_cache(self, cache: "ResponseCache | None") -> None:
        """Attach a shared LLM-call cache (no-op for deterministic judges)."""
        self._judge_cache = cache

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
