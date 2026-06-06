"""Structured-output judge: validate a target's JSON output against a JSON Schema.

Many LLM features must emit machine-readable JSON (tool calls, extraction, config
generation). This judge parses the output and validates it against a JSON Schema,
scoring 1.0 when valid and 0.0 otherwise — gateable by name like any built-in:

    judge:
      type: structured
      json_schema:                 # inline, or a path to a .json schema file
        type: object
        required: [id, name, price]
        properties:
          id:    {type: integer}
          name:  {type: string, minLength: 1}
          price: {type: number, minimum: 0}
        additionalProperties: false

    metrics:
      - {name: accuracy, threshold: 1.0, mode: absolute}

With ``partial_credit: true`` an object output instead scores the fraction of required
top-level properties that are present and individually valid — a softer "how close" signal
for tracking trends rather than a hard pass/fail.

The validator is self-contained (no extra dependency) and supports the practical subset of
JSON Schema used for structured outputs: ``type`` (incl. lists of types), ``required``,
``properties``, ``additionalProperties`` (bool), ``items``, ``enum``, ``minimum`` /
``maximum``, ``minLength`` / ``maxLength``, ``minItems`` / ``maxItems``, and ``pattern``.
"""

from __future__ import annotations

import json
import re

from llmci.judges.base import Judge
from llmci.models import JudgeResult

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _type_matches(value: object, expected: str) -> bool:
    # JSON booleans are a subtype of int in Python; keep number/integer distinct from bool.
    if expected in ("number", "integer") and isinstance(value, bool):
        return False
    py = _JSON_TYPES.get(expected)
    if py is None:
        return True  # unknown type keyword: don't penalize
    return isinstance(value, py)


def validate_schema(instance: object, schema: dict, path: str = "$") -> list[str]:
    """Validate ``instance`` against ``schema``; return a list of human-readable errors."""
    errors: list[str] = []

    expected_type = schema.get("type")
    if expected_type is not None:
        types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_type_matches(instance, t) for t in types):
            errors.append(f"{path}: expected type {expected_type}, got {_kind(instance)}")
            return errors  # further checks assume the type matched

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in enum {schema['enum']}")

    if isinstance(instance, dict):
        errors.extend(_validate_object(instance, schema, path))
    elif isinstance(instance, list):
        errors.extend(_validate_array(instance, schema, path))
    elif isinstance(instance, str):
        errors.extend(_validate_string(instance, schema, path))
    elif isinstance(instance, (int, float)) and not isinstance(instance, bool):
        errors.extend(_validate_number(instance, schema, path))

    return errors


def _validate_object(obj: dict, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    properties = schema.get("properties", {})

    for key in schema.get("required", []):
        if key not in obj:
            errors.append(f"{path}: missing required property '{key}'")

    if schema.get("additionalProperties") is False:
        extra = [k for k in obj if k not in properties]
        if extra:
            errors.append(f"{path}: unexpected properties {sorted(extra)}")

    for key, subschema in properties.items():
        if key in obj:
            errors.extend(validate_schema(obj[key], subschema, f"{path}.{key}"))

    return errors


def _validate_array(arr: list, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    if "minItems" in schema and len(arr) < schema["minItems"]:
        errors.append(f"{path}: expected >= {schema['minItems']} items, got {len(arr)}")
    if "maxItems" in schema and len(arr) > schema["maxItems"]:
        errors.append(f"{path}: expected <= {schema['maxItems']} items, got {len(arr)}")
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for i, item in enumerate(arr):
            errors.extend(validate_schema(item, item_schema, f"{path}[{i}]"))
    return errors


def _validate_string(s: str, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    if "minLength" in schema and len(s) < schema["minLength"]:
        errors.append(f"{path}: shorter than minLength {schema['minLength']}")
    if "maxLength" in schema and len(s) > schema["maxLength"]:
        errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
    pattern = schema.get("pattern")
    if pattern is not None and not re.search(pattern, s):
        errors.append(f"{path}: does not match pattern {pattern!r}")
    return errors


def _validate_number(n: float, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    if "minimum" in schema and n < schema["minimum"]:
        errors.append(f"{path}: {n} < minimum {schema['minimum']}")
    if "maximum" in schema and n > schema["maximum"]:
        errors.append(f"{path}: {n} > maximum {schema['maximum']}")
    return errors


def _kind(value: object) -> str:
    for name, py in _JSON_TYPES.items():
        if name in ("number", "integer"):
            continue
        if isinstance(value, py) and not (name != "boolean" and isinstance(value, bool)):
            return name
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


class StructuredJudge(Judge):
    """Validate JSON output against a JSON Schema (1.0 valid / 0.0 invalid)."""

    def __init__(self, schema: dict, partial_credit: bool = False):
        if not isinstance(schema, dict):
            raise ValueError("Structured judge requires a JSON Schema object")
        self.schema = schema
        self.partial_credit = partial_credit

    async def evaluate_single(
        self, input: str, expected: str, actual: str
    ) -> JudgeResult:
        try:
            instance = json.loads(actual)
        except json.JSONDecodeError as e:
            return JudgeResult(score=0.0, reason=f"invalid JSON: {e}")

        errors = validate_schema(instance, self.schema)
        if not errors:
            return JudgeResult(score=1.0)

        if self.partial_credit:
            score = self._partial_score(instance)
            return JudgeResult(score=score, reason="; ".join(errors[:5]))

        return JudgeResult(score=0.0, reason="; ".join(errors[:5]))

    def _partial_score(self, instance: object) -> float:
        """Fraction of required top-level properties present and individually valid."""
        required = self.schema.get("required", [])
        if not isinstance(instance, dict) or not required:
            return 0.0
        properties = self.schema.get("properties", {})
        passing = 0
        for key in required:
            if key not in instance:
                continue
            subschema = properties.get(key, {})
            if not validate_schema(instance[key], subschema, key):
                passing += 1
        return passing / len(required)
