"""Tests for GitHub integration."""

import json
import os
from unittest.mock import patch

from scaffold.integrations.github import (
    _extract_pr_number,
    detect_github_context,
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
