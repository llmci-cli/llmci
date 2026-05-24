"""Baseline storage: save and load eval baselines from .llmci/baselines/."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from llmci.models import EvalResult

BASELINE_DIR = Path(".llmci/baselines")


@dataclass
class Baseline:
    eval_name: str
    metrics: dict[str, float]
    timestamp: str
    commit_sha: str


def save_baseline(result: EvalResult, commit_sha: str | None = None) -> Path:
    """Write baseline to .llmci/baselines/{eval_name}.json."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    if commit_sha is None:
        commit_sha = _get_current_commit() or "unknown"

    baseline = Baseline(
        eval_name=result.eval_name,
        metrics=result.metrics,
        timestamp=datetime.now(timezone.utc).isoformat(),
        commit_sha=commit_sha,
    )

    path = BASELINE_DIR / f"{result.eval_name}.json"
    path.write_text(json.dumps(asdict(baseline), indent=2) + "\n")
    return path


def load_baseline(eval_name: str, ref: str | None = None) -> Baseline | None:
    """Load baseline for an eval.

    If ref is provided (e.g., "main", "origin/main"), load from that git ref
    using `git show {ref}:.llmci/baselines/{eval_name}.json`.

    If ref is None, load from the local filesystem.

    Returns None if no baseline exists.
    """
    if ref is not None:
        return _load_from_git_ref(eval_name, ref)
    return _load_from_disk(eval_name)


def load_all_baselines(
    eval_names: list[str], ref: str | None = None
) -> dict[str, Baseline]:
    """Load baselines for all evals. Missing baselines are omitted."""
    baselines = {}
    for name in eval_names:
        bl = load_baseline(name, ref=ref)
        if bl is not None:
            baselines[name] = bl
    return baselines


def _load_from_disk(eval_name: str) -> Baseline | None:
    """Load baseline from local .llmci/baselines/."""
    path = BASELINE_DIR / f"{eval_name}.json"
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return Baseline(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def _load_from_git_ref(eval_name: str, ref: str) -> Baseline | None:
    """Load baseline from a git ref using `git show`."""
    git_path = f"{BASELINE_DIR}/{eval_name}.json"

    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{git_path}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        return Baseline(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def _get_current_commit() -> str | None:
    """Get the current git commit SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None
