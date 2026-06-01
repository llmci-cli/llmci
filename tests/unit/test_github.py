"""Tests for GitHub integration."""

import json
import os
from unittest.mock import patch

from llmci.integrations.github import (
    COMMENT_MARKER,
    GitHubComment,
    GitHubContext,
    _delete_duplicate_comments,
    _extract_pr_number,
    _merge_existing_comment_bodies,
    _select_canonical_comment,
    build_merged_comment_body,
    detect_github_context,
    parse_report_slices,
    resolve_report_slice_key,
)


class TestDetectGitHubContext:
    def test_not_in_github_actions(self):
        with patch.dict(os.environ, {}, clear=True):
            assert detect_github_context() is None

    def test_missing_token(self):
        with patch.dict(os.environ, {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "owner/repo",
        }, clear=True):
            assert detect_github_context() is None

    def test_valid_context(self, tmp_path):
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({
            "pull_request": {"number": 42},
        }))
        with patch.dict(os.environ, {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_TOKEN": "fake-token",
            "GITHUB_EVENT_PATH": str(event_file),
        }, clear=True):
            ctx = detect_github_context()
            assert ctx is not None
            assert ctx.repository == "owner/repo"
            assert ctx.pr_number == 42
            assert ctx.token == "fake-token"


class TestExtractPrNumber:
    def test_from_pull_request_event(self, tmp_path):
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({"pull_request": {"number": 7}}))
        assert _extract_pr_number(str(event_file)) == 7

    def test_from_issue_comment_event(self, tmp_path):
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({"number": 12}))
        assert _extract_pr_number(str(event_file)) == 12

    def test_missing_file(self):
        assert _extract_pr_number("/nonexistent/event.json") is None

    def test_malformed_json(self, tmp_path):
        event_file = tmp_path / "event.json"
        event_file.write_text("not json")
        assert _extract_pr_number(str(event_file)) is None

    def test_no_pr_number(self, tmp_path):
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({"action": "push"}))
        assert _extract_pr_number(str(event_file)) is None

    def test_none_path(self):
        assert _extract_pr_number(None) is None


class TestReportSliceMerging:
    def test_build_single_slice(self):
        body = build_merged_comment_body(
            None, "json-api/llmci.yaml", "## llmci Eval Report\n\n| ok |"
        )
        assert COMMENT_MARKER in body
        assert "<!-- llmci-eval-slice:json-api/llmci.yaml -->" in body
        assert "### json-api · `llmci.yaml`" in body
        assert "## llmci Eval Report" in body

    def test_merge_second_slice(self):
        first = build_merged_comment_body(None, "json-api/llmci.yaml", "report-a")
        merged = build_merged_comment_body(first, "ticket-classifier/llmci.yaml", "report-b")
        slices = parse_report_slices(merged)
        assert slices == {
            "json-api/llmci.yaml": "report-a",
            "ticket-classifier/llmci.yaml": "report-b",
        }

    def test_update_existing_slice(self):
        initial = build_merged_comment_body(None, "json-api/llmci.yaml", "old")
        updated = build_merged_comment_body(initial, "json-api/llmci.yaml", "new")
        slices = parse_report_slices(updated)
        assert slices["json-api/llmci.yaml"] == "new"
        assert "<!-- llmci-eval-slice:ticket-classifier" not in updated

    def test_parse_legacy_comment_without_slices(self):
        legacy = f"{COMMENT_MARKER}\n## llmci Eval Report\n\nlegacy"
        assert parse_report_slices(legacy) == {}

    def test_resolve_report_slice_key_from_env(self):
        with patch.dict(os.environ, {"LLMCI_REPORT_SLICE": "rag-qa/llmci.yaml"}, clear=True):
            assert resolve_report_slice_key() == "rag-qa/llmci.yaml"

    def test_resolve_report_slice_key_empty(self):
        with patch.dict(os.environ, {"LLMCI_REPORT_SLICE": "  "}, clear=True):
            assert resolve_report_slice_key() is None

    def test_select_canonical_comment_prefers_most_slices(self):
        single = GitHubComment(
            id=1,
            body=build_merged_comment_body(None, "01-ci-regression/llmci.yaml", "report-a"),
        )
        merged = GitHubComment(
            id=2,
            body=build_merged_comment_body(
                single.body,
                "04-custom-judge/llmci.yaml",
                "report-b",
            ),
        )

        assert _select_canonical_comment([single, merged]) == merged

    def test_merge_existing_comment_bodies_preserves_slices_from_duplicates(self):
        first = GitHubComment(
            id=1,
            body=build_merged_comment_body(None, "01-ci-regression/llmci.yaml", "report-a"),
        )
        second = GitHubComment(
            id=2,
            body=build_merged_comment_body(None, "04-custom-judge/llmci.yaml", "report-b"),
        )

        merged_body = _merge_existing_comment_bodies([first, second])

        assert merged_body is not None
        assert parse_report_slices(merged_body) == {
            "01-ci-regression/llmci.yaml": "report-a",
            "04-custom-judge/llmci.yaml": "report-b",
        }

    def test_delete_duplicate_comments_keeps_canonical_id(self):
        ctx = GitHubContext(repository="owner/repo", pr_number=1, token="token")
        comments = [
            GitHubComment(id=1, body=f"{COMMENT_MARKER}\nstale"),
            GitHubComment(id=2, body=f"{COMMENT_MARKER}\ncanonical"),
        ]
        deleted: list[int] = []

        def record_deleted(comment_id: int, ctx: GitHubContext) -> bool:
            deleted.append(comment_id)
            return True

        with (
            patch("llmci.integrations.github._find_existing_comments", return_value=comments),
            patch("llmci.integrations.github._delete_comment", side_effect=record_deleted),
        ):
            _delete_duplicate_comments([2], ctx)

        assert deleted == [1]
