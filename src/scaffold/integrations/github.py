"""GitHub Actions integration: detect CI context and post PR comments."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

COMMENT_MARKER = "<!-- scaffold-eval-report -->"


@dataclass
class GitHubContext:
    """Parsed GitHub Actions environment."""

    repository: str
    pr_number: int
    token: str
    server_url: str = "https://api.github.com"


def detect_github_context() -> GitHubContext | None:
    """Read GitHub Actions environment variables.

    Returns None if not running in GitHub Actions.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return None

    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    event_path = os.environ.get("GITHUB_EVENT_PATH")

    if not repo or not token:
        return None

    pr_number = _extract_pr_number(event_path)
    if pr_number is None:
        return None

    return GitHubContext(repository=repo, pr_number=pr_number, token=token)


def post_pr_comment(report_md: str, ctx: GitHubContext) -> bool:
    """Post or update a PR comment with the eval report.

    Identifies existing Scaffold comments by the hidden marker and updates
    in place to avoid duplicate comments on re-runs.

    Returns True if the comment was posted/updated successfully.
    """
    body = f"{COMMENT_MARKER}\n{report_md}"

    existing_id = _find_existing_comment(ctx)
    if existing_id:
        return _update_comment(existing_id, body, ctx)
    else:
        return _create_comment(body, ctx)


def _extract_pr_number(event_path: str | None) -> int | None:
    """Extract PR number from the GitHub event payload."""
    if not event_path:
        return None

    try:
        event = json.loads(Path(event_path).read_text())
        pr = event.get("pull_request", {})
        number = pr.get("number")
        if number is not None:
            return int(number)

        number = event.get("number")
        if number is not None:
            return int(number)
    except (json.JSONDecodeError, OSError, ValueError):
        pass

    return None


def _find_existing_comment(ctx: GitHubContext) -> int | None:
    """Find an existing Scaffold comment on the PR."""
    url = f"{ctx.server_url}/repos/{ctx.repository}/issues/{ctx.pr_number}/comments?per_page=100"

    try:
        data = _github_api_get(url, ctx.token)
        for comment in data:
            if COMMENT_MARKER in comment.get("body", ""):
                return int(comment["id"])
    except Exception:
        pass

    return None


def _create_comment(body: str, ctx: GitHubContext) -> bool:
    """Create a new PR comment."""
    url = f"{ctx.server_url}/repos/{ctx.repository}/issues/{ctx.pr_number}/comments"
    try:
        _github_api_post(url, {"body": body}, ctx.token)
        return True
    except Exception:
        return False


def _update_comment(comment_id: int, body: str, ctx: GitHubContext) -> bool:
    """Update an existing PR comment."""
    url = f"{ctx.server_url}/repos/{ctx.repository}/issues/comments/{comment_id}"
    try:
        _github_api_patch(url, {"body": body}, ctx.token)
        return True
    except Exception:
        return False


def _github_api_get(url: str, token: str) -> list | dict:
    """Make a GET request to the GitHub API."""
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        result: list | dict = json.loads(resp.read())
        return result


def _github_api_post(url: str, data: dict, token: str) -> dict:
    """Make a POST request to the GitHub API."""
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        result: dict = json.loads(resp.read())
        return result


def _github_api_patch(url: str, data: dict, token: str) -> dict:
    """Make a PATCH request to the GitHub API."""
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="PATCH",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        result: dict = json.loads(resp.read())
        return result
