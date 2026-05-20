"""Custom judge for pipeline-level QA evaluation.

Checks whether the expected content appears in the generated answer,
allowing for paraphrasing and additional context from retrieval.
"""


def evaluate(input: str, expected: str, actual: str) -> dict:
    """Score based on whether key expected phrases appear in the answer."""
    expected_lower = expected.lower()
    actual_lower = actual.lower()

    expected_words = set(expected_lower.split())
    actual_words = set(actual_lower.split())

    overlap = expected_words & actual_words
    if not expected_words:
        return {"score": 1.0, "reason": "No expected content to check"}

    coverage = len(overlap) / len(expected_words)

    if coverage >= 0.6:
        return {
            "score": 1.0,
            "reason": f"Good coverage: {coverage:.0%} of expected terms found",
        }
    elif coverage >= 0.3:
        missing = expected_words - actual_words
        return {
            "score": 0.5,
            "reason": f"Partial coverage ({coverage:.0%}). Missing: {', '.join(sorted(missing))}",
        }
    else:
        return {
            "score": 0.0,
            "reason": f"Low coverage ({coverage:.0%}). Expected content not found in answer.",
        }
