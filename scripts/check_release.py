"""Check release metadata stays in sync."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(path: str) -> str:
    return (ROOT / path).read_text()


def find(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find {label}")
    return match.group(1)


def main() -> int:
    errors: list[str] = []

    pyproject = read("pyproject.toml")
    init_py = read("src/llmci/__init__.py")
    changelog = read("CHANGELOG.md")
    action = read("action.yml")

    try:
        pyproject_version = find(r'^version = "([^"]+)"$', pyproject, "pyproject version")
        init_version = find(r'^__version__ = "([^"]+)"$', init_py, "__version__")
        action_version_pattern = (
            r"^\s+llmci-version:\n"
            r'\s+description: "Exact llmci version to install"\n'
            r'\s+default: "([^"]+)"'
        )
        action_version = find(
            action_version_pattern,
            action,
            "action llmci-version default",
        )
    except ValueError as exc:
        print(f"release check failed: {exc}", file=sys.stderr)
        return 1

    if pyproject_version != init_version:
        errors.append(
            "Version mismatch: "
            f"pyproject.toml has {pyproject_version}, "
            f"src/llmci/__init__.py has {init_version}"
        )

    if pyproject_version != action_version:
        errors.append(
            "Version mismatch: "
            f"pyproject.toml has {pyproject_version}, action.yml installs {action_version}"
        )

    release_heading = f"## [{pyproject_version}]"
    release_link = (
        f"[{pyproject_version}]: "
        f"https://github.com/llmci-cli/llmci/releases/tag/v{pyproject_version}"
    )
    if release_heading not in changelog:
        errors.append(f"CHANGELOG.md is missing a {release_heading} section")
    if release_link not in changelog:
        errors.append(f"CHANGELOG.md is missing release link for {pyproject_version}")

    if errors:
        for error in errors:
            print(f"release check failed: {error}", file=sys.stderr)
        return 1

    print(f"Release metadata is consistent for {pyproject_version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
