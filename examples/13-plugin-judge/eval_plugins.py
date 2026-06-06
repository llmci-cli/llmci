"""Local llmci judge plugins for this example.

Listed under ``plugins:`` in ``llmci.yaml``; importing this module registers the
``json_schema`` judge so it can be referenced as ``judge: {type: json_schema}``.
"""

import json

from llmci.judges.base import Judge
from llmci.models import JudgeResult
from llmci.plugins import register_judge


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
