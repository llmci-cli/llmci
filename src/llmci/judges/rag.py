"""RAG-specific judge: faithfulness, relevance, and retrieval metrics.

A RAG pipeline (run as a command target) writes structured output JSON beyond the
answer string — typically the retrieved passages and their ids:

    {"output": "<answer>",
     "contexts": ["passage 1", "passage 2"],
     "retrieved_ids": ["doc3", "doc7"]}

Gold retrieval labels live on the dataset example under ``extra["relevant_ids"]``.

Each configured criterion produces a 0–1 sub-score that is surfaced as a gateable
aggregate metric by name (see ``metrics._compute_subscore_metrics``). The judge's
overall per-example score is the weighted mean of its criteria.
"""

from __future__ import annotations

import json

from llmci.judges import llm_cache
from llmci.judges.base import Judge
from llmci.models import EvalExample, JudgeResult, TargetResult

LLM_CRITERIA = {"faithfulness", "answer_relevance", "context_relevance"}
RETRIEVAL_CRITERIA = {"retrieval_recall", "retrieval_precision"}
SUPPORTED_CRITERIA = LLM_CRITERIA | RETRIEVAL_CRITERIA

_PROMPTS = {
    "faithfulness": (
        "You are evaluating whether an answer is grounded in the provided context.\n\n"
        "## Context\n{contexts}\n\n## Answer\n{answer}\n\n"
        "What fraction of the answer's claims are directly supported by the context? "
        "Reply with JSON only: {{\"score\": <0.0-1.0>, \"reasoning\": \"...\"}}"
    ),
    "faithfulness_decomposed": (
        "You are verifying whether each atomic claim in an answer is grounded in "
        "the provided context.\n\n"
        "## Context\n{contexts}\n\n## Answer\n{answer}\n\n"
        "List each distinct factual claim in the answer, then judge whether the "
        "context directly supports it.\n"
        "Reply with JSON only: "
        '{{"claims": [{{"text": "<claim>", "supported": true/false}}]}}'
    ),
    "answer_relevance": (
        "You are evaluating how well an answer addresses a question.\n\n"
        "## Question\n{question}\n\n## Answer\n{answer}\n\n"
        "How relevant and complete is the answer for the question? "
        "Reply with JSON only: {{\"score\": <0.0-1.0>, \"reasoning\": \"...\"}}"
    ),
    "context_relevance": (
        "You are evaluating whether retrieved context is relevant to a question.\n\n"
        "## Question\n{question}\n\n## Retrieved Context\n{contexts}\n\n"
        "What fraction of the retrieved context is relevant to answering the question? "
        "Reply with JSON only: {{\"score\": <0.0-1.0>, \"reasoning\": \"...\"}}"
    ),
}


class RagJudge(Judge):
    """Scores RAG pipeline outputs across configurable criteria."""

    def __init__(self, criteria: list[dict], model: str = "gpt-4o-mini"):
        self.model = model
        self.criteria = [self._normalize_criterion(c) for c in criteria]

    @staticmethod
    def _normalize_criterion(raw: dict) -> dict:
        name = raw.get("name") or raw.get("type")
        if not name:
            raise ValueError("RAG criterion requires a 'name' or 'type'")
        ctype = raw.get("type", name)
        if ctype not in SUPPORTED_CRITERIA:
            raise ValueError(
                f"Unknown RAG criterion type: {ctype!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_CRITERIA))}"
            )
        return {
            "name": name,
            "type": ctype,
            "weight": float(raw.get("weight", 1.0)),
            "k": raw.get("k"),
            "decompose_claims": bool(raw.get("decompose_claims", False)),
        }

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
                per_example.append(await self._evaluate_example(ex, res))
        return per_example

    async def _evaluate_example(
        self, example: EvalExample, result: TargetResult
    ) -> JudgeResult:
        contexts = _as_str_list(result.metadata.get("contexts"))
        retrieved_ids = _as_str_list(result.metadata.get("retrieved_ids"))
        relevant_ids = _as_str_list(example.extra.get("relevant_ids"))

        sub_scores: dict[str, float] = {}
        reasons: list[str] = []
        weights: list[float] = []

        for crit in self.criteria:
            score, reason = await self._score_criterion(
                crit,
                question=example.input,
                answer=result.output,
                contexts=contexts,
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
            )
            sub_scores[crit["name"]] = score
            weights.append(crit["weight"])
            if reason:
                reasons.append(f"{crit['name']}={score:.2f} ({reason})")
            else:
                reasons.append(f"{crit['name']}={score:.2f}")

        overall = _weighted_mean(list(sub_scores.values()), weights)
        return JudgeResult(
            score=overall,
            reason="; ".join(reasons) or None,
            sub_scores=sub_scores,
        )

    async def _score_criterion(
        self,
        crit: dict,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        retrieved_ids: list[str],
        relevant_ids: list[str],
    ) -> tuple[float, str | None]:
        ctype = crit["type"]

        if ctype == "retrieval_recall":
            return _retrieval_recall(retrieved_ids, relevant_ids, crit.get("k"))
        if ctype == "retrieval_precision":
            return _retrieval_precision(retrieved_ids, relevant_ids, crit.get("k"))

        # LLM-based criteria need context for grounding/relevance.
        if ctype in ("faithfulness", "context_relevance") and not contexts:
            return 0.0, "no contexts provided"

        if ctype == "faithfulness" and crit.get("decompose_claims"):
            return await self._faithfulness_decomposed(answer, contexts)

        prompt = _PROMPTS[ctype].format(
            question=question,
            answer=answer,
            contexts="\n---\n".join(contexts),
        )
        return await self._llm_score(prompt)

    async def _faithfulness_decomposed(
        self, answer: str, contexts: list[str]
    ) -> tuple[float, str | None]:
        """Per-claim faithfulness: extract claims and score each against context."""
        prompt = _PROMPTS["faithfulness_decomposed"].format(
            answer=answer,
            contexts="\n---\n".join(contexts),
        )
        try:
            content = await llm_cache.complete(
                self.model, prompt, cache=self._judge_cache
            )
            return _parse_claim_scores(content)
        except Exception as e:
            return 0.0, f"judge error: {e}"

    async def _llm_score(self, prompt: str) -> tuple[float, str | None]:
        try:
            content = await llm_cache.complete(
                self.model, prompt, cache=self._judge_cache
            )
            return _parse_score(content)
        except Exception as e:
            return 0.0, f"judge error: {e}"


def _parse_claim_scores(content: str) -> tuple[float, str | None]:
    """Parse per-claim faithfulness: fraction of claims marked supported."""
    text = content.strip()
    if text.startswith("```"):
        text = "\n".join(
            ln for ln in text.split("\n") if not ln.strip().startswith("```")
        )
    try:
        parsed = json.loads(text)
        claims = parsed.get("claims", [])
        if not isinstance(claims, list) or not claims:
            return 0.0, "no claims extracted"
        supported = 0
        unsupported: list[str] = []
        for item in claims:
            if not isinstance(item, dict):
                continue
            claim_text = str(item.get("text", "")).strip()
            if not claim_text:
                continue
            if item.get("supported"):
                supported += 1
            else:
                unsupported.append(claim_text)
        total = supported + len(unsupported)
        if total == 0:
            return 0.0, "no claims extracted"
        score = supported / total
        reason = None
        if unsupported:
            preview = "; ".join(unsupported[:3])
            extra = f" (+{len(unsupported) - 3} more)" if len(unsupported) > 3 else ""
            reason = f"{supported}/{total} claims supported; unsupported: {preview}{extra}"
        else:
            reason = f"{supported}/{total} claims supported"
        return _clamp(score), reason
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0, f"unparseable claim decomposition: {text[:120]}"


def _parse_score(content: str) -> tuple[float, str | None]:
    """Parse a {"score", "reasoning"} JSON response, tolerating code fences."""
    text = content.strip()
    if text.startswith("```"):
        text = "\n".join(
            ln for ln in text.split("\n") if not ln.strip().startswith("```")
        )
    try:
        parsed = json.loads(text)
        score = _clamp(float(parsed["score"]))
        return score, parsed.get("reasoning")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Fall back to a bare number if present.
        try:
            return _clamp(float(text)), None
        except ValueError:
            return 0.0, f"unparseable judge response: {text[:120]}"


def _retrieval_recall(
    retrieved_ids: list[str], relevant_ids: list[str], k: int | None
) -> tuple[float, str | None]:
    if not relevant_ids:
        return 1.0, "no gold relevant_ids — vacuously satisfied"
    top = retrieved_ids[:k] if k else retrieved_ids
    hits = len(set(top) & set(relevant_ids))
    return hits / len(set(relevant_ids)), f"{hits}/{len(set(relevant_ids))} relevant retrieved"


def _retrieval_precision(
    retrieved_ids: list[str], relevant_ids: list[str], k: int | None
) -> tuple[float, str | None]:
    top = retrieved_ids[:k] if k else retrieved_ids
    if not top:
        return 0.0, "no documents retrieved"
    hits = len(set(top) & set(relevant_ids))
    return hits / len(top), f"{hits}/{len(top)} retrieved are relevant"


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if not values or total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _as_str_list(value: object) -> list[str]:
    """Coerce a metadata/extra field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]
