"""Custom judge that validates JSON structure and key matching.

Expected and actual are both JSON strings. The judge checks:
1. Actual output is valid JSON
2. All keys in expected are present in actual
3. Values for matching keys are equal
"""

import json


def judge(input: str, expected: str, actual: str) -> dict:
    try:
        expected_obj = json.loads(expected)
    except json.JSONDecodeError:
        return {"score": 0.0, "reason": "Expected is not valid JSON"}

    try:
        actual_obj = json.loads(actual)
    except json.JSONDecodeError:
        return {"score": 0.0, "reason": f"Actual output is not valid JSON: {actual[:100]}"}

    if not isinstance(expected_obj, dict) or not isinstance(actual_obj, dict):
        return {"score": 1.0 if expected_obj == actual_obj else 0.0}

    expected_keys = set(expected_obj.keys())
    actual_keys = set(actual_obj.keys())

    missing_keys = expected_keys - actual_keys
    if missing_keys:
        return {
            "score": 0.0,
            "reason": f"Missing keys: {', '.join(sorted(missing_keys))}",
        }

    mismatched = []
    for key in expected_keys:
        if expected_obj[key] != actual_obj[key]:
            mismatched.append(
                f"{key}: expected {expected_obj[key]!r}, got {actual_obj[key]!r}"
            )

    if mismatched:
        return {
            "score": 0.0,
            "reason": "; ".join(mismatched),
        }

    return {"score": 1.0}
