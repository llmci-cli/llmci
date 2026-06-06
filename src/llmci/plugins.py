"""Plugin / extension API for third-party judges.

llmci ships built-in judges (``exact_match``, ``llm``, ``rag``, ``safety``, …), but
teams have domain-specific scoring logic they'd rather not fork the tool to add. This
module lets a third party register a new ``judge.type`` two ways:

1. **Installed package (entry points).** Declare an entry point in the
   ``llmci.judges`` group; its value resolves to either a ``Judge`` subclass or a
   ``(JudgeConfig) -> Judge`` factory:

       # pyproject.toml of the plugin package
       [project.entry-points."llmci.judges"]
       my_judge = "my_pkg.judges:MyJudge"

2. **Local module (config).** List dotted module paths under ``plugins:`` in
   ``llmci.yaml``; importing each module runs its top-level ``register_judge(...)``
   calls:

       plugins:
         - my_repo.eval_plugins

       evals:
         - name: my-eval
           judge: {type: my_judge}

Both paths funnel into a single registry consulted by ``judges.factory.create_judge``.
A plugin type may not shadow a built-in judge type.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import cast

from llmci.errors import ConfigError
from llmci.judges.base import Judge
from llmci.models import JudgeConfig

# A judge factory takes the eval's JudgeConfig and returns a Judge instance.
JudgeFactory = Callable[[JudgeConfig], Judge]

ENTRY_POINT_GROUP = "llmci.judges"

# Reserved names handled directly by judges.factory.create_judge; plugins can't shadow.
BUILTIN_JUDGE_TYPES = frozenset({
    "exact_match", "llm", "custom", "composite", "rag", "pairwise", "safety",
})

_JUDGE_REGISTRY: dict[str, JudgeFactory] = {}
_entry_points_loaded = False


def register_judge(type_name: str, factory: type[Judge] | JudgeFactory) -> None:
    """Register a judge type.

    ``factory`` may be a ``Judge`` subclass (instantiated with no args) or a callable
    taking the eval's ``JudgeConfig`` and returning a ``Judge``. Raises ``ConfigError``
    on an empty name, a collision with a built-in type, or a conflicting re-registration.
    """
    if not type_name:
        raise ConfigError("Judge plugin name must be non-empty")
    if type_name in BUILTIN_JUDGE_TYPES:
        raise ConfigError(
            f"Plugin judge type {type_name!r} collides with a built-in judge type"
        )

    normalized = _as_factory(factory)
    existing = _JUDGE_REGISTRY.get(type_name)
    if existing is not None and existing is not normalized:
        raise ConfigError(
            f"Judge type {type_name!r} is already registered by a different plugin"
        )
    _JUDGE_REGISTRY[type_name] = normalized


def _as_factory(factory: type[Judge] | JudgeFactory) -> JudgeFactory:
    """Normalize a Judge subclass or factory callable into a factory callable."""
    if isinstance(factory, type) and issubclass(factory, Judge):
        judge_cls = factory

        def _build(_config: JudgeConfig) -> Judge:
            return judge_cls()

        return _build
    if callable(factory):
        return cast(JudgeFactory, factory)
    raise ConfigError(
        "Judge plugin must be a Judge subclass or a (JudgeConfig) -> Judge callable"
    )


def get_judge_factory(type_name: str) -> JudgeFactory | None:
    """Return the registered factory for a judge type, or None. Loads entry points."""
    ensure_entry_points_loaded()
    return _JUDGE_REGISTRY.get(type_name)


def registered_judge_types() -> list[str]:
    """Return the sorted list of currently registered plugin judge types."""
    ensure_entry_points_loaded()
    return sorted(_JUDGE_REGISTRY)


def load_module_plugins(modules: list[str]) -> None:
    """Import dotted module paths so their top-level ``register_judge`` calls run."""
    for module_path in modules:
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            raise ConfigError(
                f"Failed to import plugin module {module_path!r}: {e}"
            ) from e


def ensure_entry_points_loaded() -> None:
    """Discover and register judges advertised via the ``llmci.judges`` entry-point group.

    Idempotent: entry points are scanned once per process. A plugin that fails to load
    is skipped with a warning rather than crashing the run.
    """
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    for ep in _select_entry_points():
        try:
            obj = ep.load()
            register_judge(ep.name, obj)
        except ConfigError:
            # Collision or bad shape — re-raise so misconfiguration is visible.
            raise
        except Exception as e:  # pragma: no cover - defensive against broken plugins
            import warnings

            warnings.warn(
                f"Failed to load llmci judge plugin {ep.name!r}: {e}",
                stacklevel=2,
            )


def _select_entry_points() -> list:
    """Return entry points in the judges group across importlib.metadata versions."""
    from importlib.metadata import entry_points

    try:
        # Python 3.10+: selectable interface.
        return list(entry_points(group=ENTRY_POINT_GROUP))
    except TypeError:  # pragma: no cover - older API fallback
        return list(entry_points().get(ENTRY_POINT_GROUP, []))


def reset_registry() -> None:
    """Clear the registry and entry-point cache. For tests only."""
    global _entry_points_loaded
    _JUDGE_REGISTRY.clear()
    _entry_points_loaded = False


__all__ = [
    "JudgeFactory", "ENTRY_POINT_GROUP", "BUILTIN_JUDGE_TYPES", "register_judge",
    "get_judge_factory", "registered_judge_types", "load_module_plugins",
    "ensure_entry_points_loaded", "reset_registry",
]
