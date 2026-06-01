"""Tests for release metadata consistency checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_check_release_module():
    module_path = Path(__file__).parent.parent.parent / "scripts" / "check_release.py"
    spec = importlib.util.spec_from_file_location("check_release", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_release_tree(
    root: Path,
    *,
    version: str = "1.2.3",
    action_version: str | None = None,
) -> None:
    action_version = action_version or version
    (root / "src" / "llmci").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    (root / "src" / "llmci" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [{version}] - 2026-05-31\n\n"
        f"[{version}]: https://github.com/llmci-cli/llmci/releases/tag/v{version}\n"
    )
    (root / "action.yml").write_text(
        "inputs:\n"
        "  llmci-version:\n"
        '    description: "Exact llmci version to install"\n'
        f'    default: "{action_version}"\n'
    )


def test_release_check_passes_when_metadata_matches(tmp_path, monkeypatch, capsys):
    module = load_check_release_module()
    write_release_tree(tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)

    assert module.main() == 0
    captured = capsys.readouterr()
    assert "Release metadata is consistent for 1.2.3." in captured.out


def test_release_check_fails_when_action_version_differs(tmp_path, monkeypatch, capsys):
    module = load_check_release_module()
    write_release_tree(tmp_path, version="1.2.3", action_version="1.2.2")
    monkeypatch.setattr(module, "ROOT", tmp_path)

    assert module.main() == 1
    captured = capsys.readouterr()
    assert "pyproject.toml has 1.2.3, action.yml installs 1.2.2" in captured.err
