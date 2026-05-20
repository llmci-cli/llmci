"""LLM-as-judge for open-ended evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import litellm

from scaffold.judges.base import Judge
from scaffold.models import EvalExample, JudgeResult, TargetResult

CACHE_DIR = Path(".scaffold/cache")

JUDGE_PROMPT_WITH_REF = """\
You are an evaluation judge. Given a user input, a reference answer, \
and an actual model output, evaluate whether the actual output satisfies \
the following criterion.

## Criterion
{criterion_prompt}

## User Input
{input}

## Reference Answer
{expected}

## Actual Output
{actual}

## Instructions
Respond with a JSON object and nothing else:
{{"passed": true, "reasoning": "brief explanation"}}
or
{{"passed": false, "reasoning": "brief explanation"}}"""

JUDGE_PROMPT_NO_REF = """\
You are an evaluation judge. Given a user input and the model's output, \
evaluate whether the output satisfies the following criterion.

## Criterion
{criterion_prompt}

## User Input
{input}

## Model Output
{actual}

## Instructions
Respond with a JSON object and nothing else:
{{"passed": true, "reasoning": "brief explanation"}}
or
{{"passed": false, "reasoning": "brief explanation"}}"""


class LLMJudge(Judge):
    """Evaluates outputs using an LLM against a rubric."""

    def __init__(
        self,
        model: str,
        rubric: list[dict] | str,
        temperature: float = 0.0,
        use_cache: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.use_cache = use_cache
        self._cache: dict[str, dict] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        if isinstance(rubric, str):
            self.rubric = [{"id": "default", "prompt": rubric}]
        else:
            self.rubric = rubric

        if use_cache:
            self._load_cache()

    async def evaluate_single(
        self, input: str, expected: str, actual: str
    ) -> JudgeResult:
        criteria_results = []

        for criterion in self.rubric:
            result = await self._evaluate_criterion(
                criterion, input, expected, actual
            )
            criteria_results.append(result)

        passed_count = sum(1 for r in criteria_results if r["passed"])
        score = passed_count / len(criteria_results) if criteria_results else 0.0

        failed = [r for r in criteria_results if not r["passed"]]
        reason = None
        if failed:
            reasons = [f"{r['id']}: {r['reasoning']}" for r in failed]
            reason = "; ".join(reasons)

        return JudgeResult(score=score, reason=reason)

    async def evaluate_dataset(
        self,
        examples: list[EvalExample],
        results: list[TargetResult],
    ) -> list[JudgeResult]:
        per_example: list[JudgeResult] = []
        for ex, res in zip(examples, results):
            if res.error is not None:
                per_example.append(
                    JudgeResult(score=0.0, reason=f"Target error: {res.error}")
                )
            else:
                result = await self.evaluate_single(ex.input, ex.expected, res.output)
                per_example.append(result)

        if self.use_cache:
            self._save_cache()

        return per_example

    async def _evaluate_criterion(
        self,
        criterion: dict,
        input: str,
        expected: str,
        actual: str,
    ) -> dict:
        """Evaluate a single criterion. Returns dict with id, passed, reasoning."""
        criterion_id = criterion.get("id", "unknown")
        criterion_prompt = criterion.get("prompt", "")

        cache_key = self._cache_key(criterion_prompt, input, expected, actual)
        if self.use_cache and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1

        if expected:
            prompt = JUDGE_PROMPT_WITH_REF.format(
                criterion_prompt=criterion_prompt,
                input=input,
                expected=expected,
                actual=actual,
            )
        else:
            prompt = JUDGE_PROMPT_NO_REF.format(
                criterion_prompt=criterion_prompt,
                input=input,
                actual=actual,
            )

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                timeout=30,
            )
            content = response.choices[0].message.content or ""
            parsed = self._parse_response(content)
            result = {
                "id": criterion_id,
                "passed": parsed["passed"],
                "reasoning": parsed.get("reasoning", ""),
            }
        except Exception as e:
            result = {
                "id": criterion_id,
                "passed": False,
                "reasoning": f"Judge error: {e}",
            }

        if self.use_cache:
            self._cache[cache_key] = result

        return result

    def _parse_response(self, content: str) -> dict:
        """Parse the JSON response from the judge LLM."""
        content = content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            parsed: dict = json.loads(content)
            if "passed" not in parsed:
                return {"passed": False, "reasoning": "Missing 'passed' field in response"}
            return parsed
        except json.JSONDecodeError:
            lower = content.lower()
            if '"passed": true' in lower or '"passed":true' in lower:
                return {"passed": True, "reasoning": content[:200]}
            return {"passed": False, "reasoning": f"Could not parse response: {content[:200]}"}

    def _cache_key(
        self, criterion: str, input: str, expected: str, actual: str
    ) -> str:
        """Deterministic cache key from inputs."""
        data = f"{self.model}|{criterion}|{input}|{expected}|{actual}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _load_cache(self) -> None:
        """Load judge result cache from disk."""
        cache_file = CACHE_DIR / "judge_results.json"
        if cache_file.exists():
            try:
                self._cache = json.loads(cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _save_cache(self) -> None:
        """Persist judge result cache to disk."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / "judge_results.json"
        try:
            cache_file.write_text(json.dumps(self._cache, indent=2))
        except OSError:
            pass

    @property
    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._cache_hits, "misses": self._cache_misses}
