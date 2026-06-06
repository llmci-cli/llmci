"""Local llmci judge and metric plugins for this example.

Listed under ``plugins:`` in ``llmci.yaml``; importing this module registers the
``json_schema`` judge (``judge: {type: json_schema}``) and the ``json_field_coverage``
metric (gateable by name like any built-in metric).
"""

import json

from llmci.judges.base import Judge
from llmci.models import JudgeResult
from llmci.plugins import MetricContext, register_judge, register_metric


class JsonSchemaJudge(Judge):
    """Score 1.0 if the output is valid JSON containing the expected keys.

    The dataset's ``expected`` field is a comma-separated list of required top-level
    keys (e.g. ``"id,name,price"``).
    """

    async def evaluate_single(
        self, input: str, expected: str, actual: str
    ) -> JudgeResult:
        try:
            obj = json.loads(actual)
        except json.JSONDecodeError as e:
            return JudgeResult(score=0.0, reason=f"invalid JSON: {e}")

        if not isinstance(obj, dict):
            return JudgeResult(score=0.0, reason="output is not a JSON object")

        required = [k.strip() for k in expected.split(",") if k.strip()]
        missing = [k for k in required if k not in obj]
        if missing:
            return JudgeResult(
                score=0.0, reason=f"missing keys: {', '.join(missing)}"
            )
        return JudgeResult(score=1.0)


register_judge("json_schema", JsonSchemaJudge)


def json_field_coverage(ctx: MetricContext) -> float:
    """Mean fraction of required keys present across valid examples.

    A softer companion to the pass/fail judge: even a response missing one key still
    contributes partial credit, so you can track coverage trends, not just pass rate.
    """
    fractions: list[float] = []
    for i in ctx.valid_indices:
        required = [k.strip() for k in ctx.examples[i].expected.split(",") if k.strip()]
        if not required:
            continue
        try:
            obj = json.loads(ctx.results[i].output)
        except json.JSONDecodeError:
            fractions.append(0.0)
            continue
        present = sum(1 for k in required if isinstance(obj, dict) and k in obj)
        fractions.append(present / len(required))
    return sum(fractions) / len(fractions) if fractions else 0.0


register_metric("json_field_coverage", json_field_coverage)
