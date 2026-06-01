# Releasing llmci

Install from PyPI as **`llmci`**. The CLI command is **`llmci`**.

Do not use `pip install scaffold-ai` — that name is taken by an unrelated project-scaffolding tool.

PyPI project: https://pypi.org/project/llmci/

## One-time setup

### PyPI trusted publisher

After the first manual upload (or if trusted publishing is not configured yet):

1. PyPI → [`llmci`](https://pypi.org/manage/project/llmci/settings/publishing/) → **Publishing** → **Add trusted publisher**
2. Set:
   - **Owner:** `llmci-cli`
   - **Repository:** `llmci`
   - **Workflow:** `publish.yml`
   - **Environment:** `pypi`
3. No API token is required in GitHub once this is configured.

### GitHub environment

In `llmci-cli/llmci`:

1. **Settings → Environments → New environment** → name it `pypi`
2. Leave **secrets empty** — trusted publishing uses OIDC, not a `PYPI_API_TOKEN`
3. Optional: restrict deployment to tag pattern `v*`
4. Optional: add **Required reviewers** for a manual approval gate before publish

### Publish workflow

Releases are published by [`.github/workflows/publish.yml`](.github/workflows/publish.yml) when a GitHub release is **published** (not drafted).

The publish job must include both permissions:

```yaml
permissions:
  contents: read   # required for actions/checkout
  id-token: write  # required for PyPI trusted publishing
```

Job-level `permissions` replace workflow-level permissions entirely. If `contents: read` is missing, checkout fails with `Repository not found`.

## Every release

### 1. Preflight on `main`

```bash
ruff check src/ tests/
mypy src/
pytest tests/ -v
python scripts/check_release.py
```

Wait for the **llmci Dogfood** workflow to pass on `main` before tagging.

### 2. Bump version

Update both:

- `pyproject.toml` → `[project].version`
- `src/llmci/__init__.py` → `__version__`

Use semver. Stay on `0.x` while iterating quickly.

### 3. Update changelog

In `CHANGELOG.md`:

1. Move `[Unreleased]` notes into a new `## [X.Y.Z] - YYYY-MM-DD` section
2. Leave an empty `## [Unreleased]` section at the top
3. Add a release link at the bottom, e.g. `[0.1.5]: https://github.com/llmci-cli/llmci/releases/tag/v0.1.5`

### 4. Commit, tag, and push

```bash
git add CHANGELOG.md pyproject.toml src/llmci/__init__.py
git commit -m "Release llmci X.Y.Z."
git tag vX.Y.Z
git push origin main
git push origin vX.Y.Z
```

If HTTPS push fails locally, use SSH:

```bash
git push git@github.com:llmci-cli/llmci.git main
git push git@github.com:llmci-cli/llmci.git vX.Y.Z
```

### 5. Publish the GitHub release

Publishing the release triggers the PyPI workflow. A draft release does **not** publish.

**GitHub UI:** **Releases → Draft new release** → pick tag `vX.Y.Z` → **Publish release**

**CLI:**

```bash
gh release create vX.Y.Z \
  --repo llmci-cli/llmci \
  --title "llmci X.Y.Z" \
  --notes "$(cat <<'EOF'
## Added / Changed / Fixed
- ...

**Install:** `pip install llmci==X.Y.Z`
EOF
)"
```

### 6. Verify

1. **Actions → Publish to PyPI** — job should succeed
2. PyPI shows the new version: https://pypi.org/project/llmci/
3. Local smoke check:

```bash
pip install 'llmci==X.Y.Z'
llmci --version
```

## After release

| Location | What to update |
|----------|----------------|
| README / docs | `pip install llmci` or pin `llmci==X.Y.Z` in examples |
| `action.yml` | `llmci-version` default |
| `llmci-testbed` CI | `pip install --upgrade llmci` |
| `llmci-testbed` `pyproject.toml` | `llmci>=X.Y.Z` |

Pin in production: `llmci==X.Y.Z`. Use `--upgrade` only in dogfood / testbed CI.

During active development before a PyPI cut:

```bash
pip install "llmci @ git+https://github.com/llmci-cli/llmci@main"
```

## Manual publish (fallback)

Use only if the GitHub publish workflow is broken.

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
twine upload dist/*
```

PyPI will prompt for credentials. Use an API token scoped to `llmci` (Account settings → API tokens).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Checkout: `Repository not found` | Publish job missing `contents: read` | Add it to the job's `permissions` |
| Re-run of old publish job still fails | Re-runs use the workflow from the original commit | Cut a new patch tag and release |
| PyPI publish succeeds but version unchanged | GitHub release was left as draft | Publish the release |
| S3 tests fail in CI without boto3 | Tests must mock `boto3`, not import it | See `tests/unit/test_dataset_remote.py` |
| `git push` over HTTPS fails locally | Remote uses HTTPS; `gh` auth is SSH | Push via `git@github.com:llmci-cli/llmci.git` or update `origin` |

## Pre-release checklist

- [ ] `ruff check src/ tests/` clean
- [ ] `mypy src/` clean
- [ ] `pytest tests/ -v` green
- [ ] `python scripts/check_release.py` clean
- [ ] Dogfood workflow green on `main`
- [ ] Version bumped in `pyproject.toml`, `src/llmci/__init__.py`, and `action.yml`
- [ ] `CHANGELOG.md` updated with release date and link
- [ ] Tag pushed: `vX.Y.Z`
- [ ] GitHub release **published** (not draft)
- [ ] Publish to PyPI workflow succeeded
- [ ] `pip install llmci==X.Y.Z` works
