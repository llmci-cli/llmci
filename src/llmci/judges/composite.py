"""Composite judge for agent evaluation — combines outcome, constraint, and trajectory judges."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from llmci.judges.base import Judge

if TYPE_CHECKING:
    from llmci.cache import ResponseCache
from llmci.models import (
    AgentConstraints,
    AgentScenario,
    AgentTrace,
    JudgeResult,
)


class ConstraintJudge:
    """Deterministic check against tool/token budgets."""

    def evaluate(
        self, trace: AgentTrace, constraints: AgentConstraints
    ) -> JudgeResult:
        """Score = fraction of constraints satisfied."""
        checks: list[tuple[str, bool]] = []

        if constraints.max_tool_calls is not None:
            passed = trace.total_tool_calls <= constraints.max_tool_calls
            checks.append((
                f"tool_calls: {trace.total_tool_calls} <= {constraints.max_tool_calls}",
                passed,
            ))

        if constraints.max_tokens is not None:
            passed = trace.total_tokens <= constraints.max_tokens
            checks.append((
                f"tokens: {trace.total_tokens} <= {constraints.max_tokens}",
                passed,
            ))

        if constraints.required_tools:
            used_tools = {s.tool for s in trace.trace if s.tool}
            for tool in constraints.required_tools:
                present = tool in used_tools
                checks.append((f"required tool '{tool}'", present))

        if constraints.forbidden_tools:
            used_tools = {s.tool for s in trace.trace if s.tool}
            for tool in constraints.forbidden_tools:
                absent = tool not in used_tools
                checks.append((f"forbidden tool '{tool}' absent", absent))

        if not checks:
            return JudgeResult(score=1.0, reason="No constraints defined")

        passed_count = sum(1 for _, ok in checks if ok)
        score = passed_count / len(checks)

        failed = [desc for desc, ok in checks if not ok]
        reason = (
            "All constraints satisfied"
            if not failed
            else f"Failed: {'; '.join(failed)}"
        )

        return JudgeResult(score=score, reason=reason)


class OutcomeJudge:
    """LLM-based evaluation of final outcome correctness."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    async def evaluate(
        self,
        scenario: AgentScenario,
        trace: AgentTrace,
        *,
        cache: "ResponseCache | None" = None,
    ) -> JudgeResult:
        """Ask an LLM to evaluate the final output against expected outcome."""
        from llmci.judges import llm_cache

        expected = (
            scenario.expected.outcome
            if scenario.expected
            else "No expected outcome defined"
        )

        prompt = (
            "You are evaluating an AI agent's output.\n\n"
            f"## Expected Outcome\n{expected}\n\n"
            f"## Agent Output\n{trace.final_output or '(no output)'}\n\n"
            "Rate the output on a scale of 0.0 to 1.0 where:\n"
            "- 1.0 = perfectly matches the expected outcome\n"
            "- 0.5 = partially correct\n"
            "- 0.0 = completely wrong\n\n"
            'Respond with JSON: {"score": <float>, "reason": "<explanation>"}'
        )

        try:
            content = await llm_cache.complete(
                self.model, prompt, cache=cache, temperature=0.0, timeout=30
            )
            return _parse_judge_response(content)
        except Exception as e:
            return JudgeResult(score=0.0, reason=f"Outcome judge error: {e}")


class TrajectoryJudge:
    """LLM-based evaluation of execution path quality."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    async def evaluate(
        self,
        trace: AgentTrace,
        rubric: str,
        *,
        cache: "ResponseCache | None" = None,
    ) -> JudgeResult:
        """Ask an LLM to evaluate the trace against a trajectory rubric."""
        from llmci.judges import llm_cache

        trace_text = _format_trace(trace)

        prompt = (
            "You are evaluating the execution trajectory of an AI agent.\n\n"
            f"## Evaluation Rubric\n{rubric}\n\n"
            f"## Agent Execution Trace\n{trace_text}\n\n"
            "Rate the trajectory on a scale of 0.0 to 1.0 where:\n"
            "- 1.0 = optimal execution path\n"
            "- 0.5 = reasonable but suboptimal\n"
            "- 0.0 = poor execution path\n\n"
            'Respond with JSON: {"score": <float>, "reason": "<explanation>"}'
        )

        try:
            content = await llm_cache.complete(
                self.model, prompt, cache=cache, temperature=0.0, timeout=30
            )
            return _parse_judge_response(content)
        except Exception as e:
            return JudgeResult(score=0.0, reason=f"Trajectory judge error: {e}")


class CompositeAgentJudge(Judge):
    """Weighted combination of outcome, constraint, and trajectory judges."""

    def __init__(self, criteria: list[dict], model: str = "gpt-4o-mini"):
        self.criteria = criteria
        self.model = model

    async def evaluate_scenario(
        self, scenario: AgentScenario, trace: AgentTrace
    ) -> JudgeResult:
        """Run each criterion judge, return weighted composite result."""
        results: list[tuple[str, float, JudgeResult]] = []

        for criterion in self.criteria:
            name = criterion.get("name", "unnamed")
            ctype = criterion.get("type", "constraint")
            weight = criterion.get("weight", 1.0)

            if ctype == "outcome":
                judge = OutcomeJudge(model=self.model)
                result = await judge.evaluate(scenario, trace, cache=self._judge_cache)

            elif ctype == "trajectory":
                rubric = criterion.get("rubric", "Evaluate the execution quality.")
                judge_t = TrajectoryJudge(model=self.model)
                result = await judge_t.evaluate(trace, rubric, cache=self._judge_cache)

            elif ctype == "constraint":
                constraints = _get_constraints(scenario, criterion)
                judge_c = ConstraintJudge()
                result = judge_c.evaluate(trace, constraints)

            else:
                result = JudgeResult(score=0.0, reason=f"Unknown criterion type: {ctype}")

            results.append((name, weight, result))

        total_weight = sum(w for _, w, _ in results)
        if total_weight == 0:
            return JudgeResult(score=0.0, reason="No criteria defined")

        composite_score = sum(w * r.score for _, w, r in results) / total_weight

        details = "; ".join(
            f"{name}={r.score:.2f} (w={w})" for name, w, r in results
        )

        return JudgeResult(score=composite_score, reason=details)


def _get_constraints(
    scenario: AgentScenario, criterion: dict
) -> AgentConstraints:
    """Extract constraints from the scenario or criterion config."""
    if scenario.expected and scenario.expected.constraints:
        return scenario.expected.constraints
    if scenario.conversation_constraints:
        return scenario.conversation_constraints

    return AgentConstraints(
        max_tool_calls=criterion.get("max_tool_calls"),
        required_tools=criterion.get("required_tools"),
        forbidden_tools=criterion.get("forbidden_tools"),
        max_tokens=criterion.get("max_tokens"),
    )


def _parse_judge_response(content: str) -> JudgeResult:
    """Parse a JSON judge response."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(
            ln for ln in lines if not ln.strip().startswith("```")
        )

    try:
        parsed: dict = json.loads(content)
        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reason = str(parsed.get("reason", ""))
        return JudgeResult(score=score, reason=reason)
    except (json.JSONDecodeError, ValueError, TypeError):
        return JudgeResult(score=0.0, reason=f"Could not parse judge response: {content[:200]}")


def _format_trace(trace: AgentTrace) -> str:
    """Format a trace for LLM consumption."""
    lines = []
    for step in trace.trace:
        if step.type == "tool_call":
            args_str = json.dumps(step.args) if step.args else ""
            lines.append(f"[Step {step.step}] TOOL CALL: {step.tool}({args_str})")
        else:
            lines.append(f"[Step {step.step}] RESPONSE: {step.content or ''}")

    if trace.final_output:
        lines.append(f"\nFinal Output: {trace.final_output}")

    lines.append(f"\nTotal tool calls: {trace.total_tool_calls}")
    lines.append(f"Total tokens: {trace.total_tokens}")

    return "\n".join(lines)
