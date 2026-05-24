# PyPI publishing — `llmci`

Scaffold installs from PyPI as **`llmci`**. The CLI command remains **`scaffold`**.

Do not use `pip install scaffold-ai` — that name is taken by an unrelated project-scaffolding tool.

## Pre-publish checklist

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Lint clean: `ruff check src/ tests/`
- [ ] Version bumped in `pyproject.toml` (semver; stay `0.x` while iterating)
- [ ] `CHANGELOG` or GitHub release notes drafted for the version
- [ ] README / docs / `action.yml` reference `llmci`
- [ ] PyPI account created at https://pypi.org
- [ ] Package name reserved: https://pypi.org/project/llmci/ (should 404 before first publish)

## First publish (manual)

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
twine upload dist/*
```

PyPI will prompt for credentials. Use an API token (Account settings → API tokens → scope to `llmci`).

Verify:

```bash
pip install llmci
scaffold --version
```

## Trusted publishing (recommended for CI)

After the first manual upload, enable [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/):

1. PyPI → `llmci` → Publishing → Add trusted publisher
2. Owner: `alexminnaar`, repo: `scaffold`, workflow: `publish.yml`, environment: `pypi`
3. Add `.github/workflows/publish.yml` triggered on `release: published` or tag push `v*`
4. GitHub repo → Settings → Environments → `pypi` (optional approval gate)

Example workflow (add when ready):

```yaml
name: Publish to PyPI
on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

## Version bumps

1. Merge changes to `main`
2. Bump `version` in `pyproject.toml`
3. Tag: `git tag v0.1.1 && git push origin v0.1.1`
4. Create GitHub release from tag → triggers publish workflow

## Downstream updates after publish

| Location | Install line |
|----------|--------------|
| README / docs | `pip install llmci` |
| `action.yml` | `pip install llmci` |
| testbed CI | `pip install --upgrade llmci` |
| testbed `pyproject.toml` | `llmci>=0.1.2` |

Pin in production: `llmci==0.1.0`. Use `--upgrade` only in CI dogfood jobs.

## Git vs PyPI during active development

Until `0.1.0` is on PyPI, install from GitHub:

```bash
pip install "llmci @ git+https://github.com/alexminnaar/scaffold@main"
```

After publish, switch testbed and examples to `pip install llmci`.
