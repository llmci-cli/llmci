"""GitHub Actions integration: detect CI context and post PR comments."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

COMMENT_MARKER = "<!-- llmci-eval-report -->"
SLICE_MARKER_PREFIX = "<!-- llmci-eval-slice:"
SLICE_MARKER_SUFFIX = " -->"
SLICE_PATTERN = re.compile(
    r"<!-- llmci-eval-slice:([^>]+) -->\n",
    re.MULTILINE,
)
MERGE_MAX_ATTEMPTS = 5


@dataclass
class GitHubComment:
    """Existing GitHub issue comment containing an llmci report."""

    id: int
    body: str


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


def resolve_report_slice_key() -> str | None:
    """Return CI slice key for merged PR comments, if set."""
    value = os.environ.get("LLMCI_REPORT_SLICE", "").strip()
    return value or None


def post_pr_comment(
    report_md: str,
    ctx: GitHubContext,
    slice_key: str | None = None,
) -> bool:
    """Post or update a PR comment with the eval report.

    When ``slice_key`` is set, merges this run's report into a shared comment
    (for CI matrix jobs). Each slice is keyed by ``LLMCI_REPORT_SLICE``.
    Without a slice key, replaces the entire comment body (single-job CI).

    Returns True if the comment was posted/updated successfully.
    """
    if slice_key:
        return _post_merged_slice(report_md, ctx, slice_key)

    body = f"{COMMENT_MARKER}\n{report_md}"
    existing = _find_existing_comment(ctx)
    if existing:
        if not _update_comment(existing.id, body, ctx):
            return False
        _delete_duplicate_comments([existing.id], ctx)
        return True
    return _create_comment(body, ctx)


def build_merged_comment_body(
    existing_body: str | None,
    slice_key: str,
    report_md: str,
) -> str:
    """Build a PR comment body with one updated report slice."""
    slices = parse_report_slices(existing_body or "")
    slices[slice_key] = report_md.strip()

    parts = [COMMENT_MARKER, ""]
    for key in sorted(slices.keys()):
        parts.append(_slice_marker(key))
        parts.append(_slice_heading(key).rstrip())
        parts.append(slices[key])
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def parse_report_slices(body: str) -> dict[str, str]:
    """Parse slice key -> report markdown from an existing PR comment."""
    slices: dict[str, str] = {}
    if COMMENT_MARKER not in body:
        return slices

    matches = list(SLICE_PATTERN.finditer(body))
    if not matches:
        return slices

    for index, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        slices[key] = _strip_slice_heading(content)
    return slices


def _post_merged_slice(report_md: str, ctx: GitHubContext, slice_key: str) -> bool:
    """Fetch-merge-update loop to reduce lost slices from parallel matrix jobs."""
    for attempt in range(MERGE_MAX_ATTEMPTS):
        existing_comments = _find_existing_comments(ctx)
        existing = _select_canonical_comment(existing_comments)
        if existing:
            existing_body = _merge_existing_comment_bodies(existing_comments)
            body = build_merged_comment_body(existing_body, slice_key, report_md)
            if not _update_comment(existing.id, body, ctx):
                time.sleep(0.2 * (attempt + 1))
                continue
        else:
            body = build_merged_comment_body(None, slice_key, report_md)
            if not _create_comment(body, ctx):
                time.sleep(0.2 * (attempt + 1))
                continue

        verified = _find_existing_comment(ctx)
        if verified and _slice_present(verified.body, slice_key, report_md):
            _delete_duplicate_comments([verified.id], ctx)
            return True

        time.sleep(0.2 * (attempt + 1))

    return False


def _merge_existing_comment_bodies(comments: list[GitHubComment]) -> str | None:
    """Merge report slices from duplicate comments into one synthetic body."""
    if not comments:
        return None

    merged_slices: dict[str, str] = {}
    for comment in comments:
        merged_slices.update(parse_report_slices(comment.body))

    if not merged_slices:
        canonical = _select_canonical_comment(comments)
        return canonical.body if canonical else None

    parts = [COMMENT_MARKER, ""]
    for key in sorted(merged_slices.keys()):
        parts.append(_slice_marker(key))
        parts.append(_slice_heading(key).rstrip())
        parts.append(merged_slices[key])
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _slice_present(body: str, slice_key: str, report_md: str) -> bool:
    slices = parse_report_slices(body)
    if slice_key not in slices:
        return False
    return slices[slice_key].strip() == report_md.strip()


def _slice_marker(key: str) -> str:
    return f"{SLICE_MARKER_PREFIX}{key}{SLICE_MARKER_SUFFIX}"


def _slice_heading(key: str) -> str:
    if "/" in key:
        service, config = key.split("/", 1)
        return f"### {service} · `{config}`\n"
    return f"### {key}\n"


def _strip_slice_heading(content: str) -> str:
    lines = content.splitlines()
    if lines and lines[0].startswith("### "):
        return "\n".join(lines[1:]).strip()
    return content


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


def _find_existing_comment(ctx: GitHubContext) -> GitHubComment | None:
    """Find an existing llmci comment on the PR."""
    comments = _find_existing_comments(ctx)
    return _select_canonical_comment(comments)


def _find_existing_comments(ctx: GitHubContext) -> list[GitHubComment]:
    """Find all existing llmci comments on the PR."""
    url = f"{ctx.server_url}/repos/{ctx.repository}/issues/{ctx.pr_number}/comments?per_page=100"

    try:
        data = _github_api_get(url, ctx.token)
        comments = []
        for comment in data:
            body = comment.get("body", "")
            if COMMENT_MARKER in body:
                comments.append(GitHubComment(id=int(comment["id"]), body=body))
        return comments
    except Exception:
        pass

    return []


def _select_canonical_comment(comments: list[GitHubComment]) -> GitHubComment | None:
    """Choose the comment to keep when duplicate llmci comments exist."""
    if not comments:
        return None
    return max(comments, key=lambda comment: len(parse_report_slices(comment.body)))


def _delete_duplicate_comments(keep_ids: list[int], ctx: GitHubContext) -> None:
    """Best-effort cleanup for stale duplicate llmci PR comments."""
    keep = set(keep_ids)
    for comment in _find_existing_comments(ctx):
        if comment.id not in keep:
            _delete_comment(comment.id, ctx)


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


def _delete_comment(comment_id: int, ctx: GitHubContext) -> bool:
    """Delete an existing PR comment."""
    url = f"{ctx.server_url}/repos/{ctx.repository}/issues/comments/{comment_id}"
    try:
        _github_api_delete(url, ctx.token)
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


def _github_api_delete(url: str, token: str) -> None:
    """Make a DELETE request to the GitHub API."""
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
        method="DELETE",
    )
    with urllib.request.urlopen(req, timeout=15):
        return None
