"""Pairwise / preference judge — compares the current output against the baseline.

Absolute scoring is weak for open-ended generation ("is this answer good?" is hard
to calibrate). Pairwise judging instead asks the easier question "is the new answer
better than the previous one?" for the same input, producing a **win rate** vs the
baseline.

The baseline output for each example comes from the stored baseline (see
``baseline.BaselineExample``), so this judge needs a baseline to compare against —
it is set on the judge by the runner before evaluation. Examples with no baseline
output (e.g. newly added rows) score a neutral 0.5.

Per-example results carry a ``win_rate`` sub-score (1.0 win / 0.5 tie / 0.0 loss),
which is surfaced as a gateable aggregate metric by the same mechanism RAG uses.
"""

from __future__ import annotations

import json

import litellm

from llmci.baseline import Baseline
from llmci.judges.base import Judge
from llmci.models import EvalExample, JudgeResult, TargetResult

DEFAULT_CRITERION = "Which answer is more correct, helpful, and complete?"

_PROMPT = """\
You are comparing two answers to the same question and deciding which is better.

## Criterion
{criterion}

## Question
{question}

## Answer A
{answer_a}

## Answer B
{answer_b}

Decide which answer is better by the criterion. Reply with JSON only:
{{"winner": "A" | "B" | "tie", "reasoning": "brief explanation"}}"""


class PairwiseJudge(Judge):
    """Scores each current output against the baseline output as a win/tie/loss."""

    def __init__(self, model: str = "gpt-4o-mini", criterion: str | None = None):
        self.model = model
        self.criterion = criterion or DEFAULT_CRITERION
        self.baseline_outputs: dict[str, str] = {}

    def set_baseline(self, baseline: Baseline | None) -> None:
        """Provide the baseline outputs to compare against (called by the runner)."""
        if baseline is None:
            self.baseline_outputs = {}
        else:
            self.baseline_outputs = {e.input: e.output for e in baseline.examples}

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
                continue

            baseline_output = self.baseline_outputs.get(ex.input)
            if baseline_output is None:
                per_example.append(JudgeResult(
                    score=0.5,
                    reason="no baseline output to compare (new example)",
                    sub_scores={"win_rate": 0.5},
                ))
                continue

            score, reason = await self._compare(ex.input, baseline_output, res.output)
            per_example.append(
                JudgeResult(score=score, reason=reason, sub_scores={"win_rate": score})
            )
        return per_example

    async def _compare(
        self, question: str, baseline_output: str, current_output: str
    ) -> tuple[float, str]:
        """Return (score, reason). Answer A = baseline, Answer B = current."""
        prompt = _PROMPT.format(
            criterion=self.criterion,
            question=question,
            answer_a=baseline_output,
            answer_b=current_output,
        )
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=30,
            )
            content = response.choices[0].message.content or ""
            return _parse_winner(content)
        except Exception as e:
            return 0.5, f"judge error: {e}"


def _parse_winner(content: str) -> tuple[float, str]:
    """Parse a {"winner", "reasoning"} response. B (current) winning scores 1.0."""
    text = content.strip()
    if text.startswith("```"):
        text = "\n".join(
            ln for ln in text.split("\n") if not ln.strip().startswith("```")
        )
    try:
        parsed = json.loads(text)
        winner = str(parsed.get("winner", "")).strip().lower()
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, TypeError):
        lowered = text.lower()
        if '"winner": "b"' in lowered or "winner: b" in lowered:
            winner, reasoning = "b", text[:120]
        elif '"winner": "a"' in lowered or "winner: a" in lowered:
            winner, reasoning = "a", text[:120]
        else:
            return 0.5, f"unparseable judge response: {text[:120]}"

    if winner == "b":
        return 1.0, f"current preferred: {reasoning}"
    if winner == "a":
        return 0.0, f"baseline preferred: {reasoning}"
    return 0.5, f"tie: {reasoning}"
