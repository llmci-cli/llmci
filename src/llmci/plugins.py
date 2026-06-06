"""Plugin / extension API for third-party judges and metrics.

llmci ships built-in judges and metrics, but teams have domain-specific scoring logic
they'd rather not fork the tool to add. This module lets a third party register a new
``judge.type`` or a custom metric by name, two ways:

1. **Installed package (entry points).** Declare an entry point in the ``llmci.judges``
   or ``llmci.metrics`` group; a judge value resolves to a ``Judge`` subclass or a
   ``(JudgeConfig) -> Judge`` factory, and a metric value to a ``(MetricContext) ->
   float`` callable:

       # pyproject.toml of the plugin package
       [project.entry-points."llmci.judges"]
       my_judge = "my_pkg.judges:MyJudge"
       [project.entry-points."llmci.metrics"]
       my_metric = "my_pkg.metrics:my_metric"

2. **Local module (config).** List dotted module paths under ``plugins:`` in
   ``llmci.yaml``; importing each module runs its top-level ``register_judge(...)`` /
   ``register_metric(...)`` calls:

       plugins:
         - my_repo.eval_plugins

       evals:
         - name: my-eval
           judge: {type: my_judge}
           metrics:
             - {name: my_metric, threshold: 0.9, mode: absolute}

Judges funnel into a registry consulted by ``judges.factory.create_judge``; metrics into
one consulted by ``metrics.compute_metrics``. Plugin names may not shadow built-ins.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from llmci.errors import ConfigError
from llmci.judges.base import Judge
from llmci.models import (
    EvalConfig,
    EvalExample,
    EvalResult,
    JudgeConfig,
    JudgeResult,
    TargetResult,
)

# A judge factory takes the eval's JudgeConfig and returns a Judge instance.
JudgeFactory = Callable[[JudgeConfig], Judge]

ENTRY_POINT_GROUP = "llmci.judges"
ENTRY_POINT_GROUP_METRICS = "llmci.metrics"
ENTRY_POINT_GROUP_REPORTERS = "llmci.reporters"

# Reserved names handled directly by judges.factory.create_judge; plugins can't shadow.
BUILTIN_JUDGE_TYPES = frozenset({
    "exact_match", "llm", "custom", "composite", "rag", "pairwise", "safety",
})

_JUDGE_REGISTRY: dict[str, JudgeFactory] = {}
_entry_points_loaded = False


@dataclass
class MetricContext:
    """Inputs available to a custom metric function.

    ``valid_indices`` are the positions of examples whose target did not error;
    ``scores`` are the judge scores at those positions (a convenience for the common
    case of aggregating over successful examples).
    """

    examples: list[EvalExample]
    results: list[TargetResult]
    per_example: list[JudgeResult]
    valid_indices: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


# A metric function reduces a MetricContext to a single aggregate value.
MetricFn = Callable[["MetricContext"], float]

_METRIC_REGISTRY: dict[str, MetricFn] = {}
_METRIC_LOWER_IS_BETTER: set[str] = set()


@dataclass
class ReportContext:
    """Inputs handed to a report sink after an eval run completes.

    ``report_markdown`` is the canonical rendered report; ``passed`` is the overall gate
    result. ``results``/``configs`` are the raw materials for a sink that wants to render
    its own format (e.g. post a Slack summary or upload an artifact).
    """

    results: list[EvalResult]
    configs: list[EvalConfig]
    passed: bool
    report_markdown: str


# A report sink consumes a ReportContext for its side effect (post, upload, notify).
ReporterFn = Callable[["ReportContext"], None]

_REPORTER_REGISTRY: dict[str, ReporterFn] = {}


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


def register_metric(
    name: str, fn: MetricFn, *, lower_is_better: bool = False
) -> None:
    """Register a custom metric computed from a :class:`MetricContext`.

    ``lower_is_better`` flips threshold direction (like cost/latency) so an `absolute`
    gate passes when the value is ``<=`` the threshold. Raises ``ConfigError`` on an
    empty name, a collision with a built-in metric, or a conflicting re-registration.
    """
    if not name:
        raise ConfigError("Metric plugin name must be non-empty")

    from llmci.metrics import BUILTIN_METRIC_NAMES

    if name in BUILTIN_METRIC_NAMES:
        raise ConfigError(
            f"Plugin metric {name!r} collides with a built-in metric"
        )
    if not callable(fn):
        raise ConfigError("Metric plugin must be a (MetricContext) -> float callable")

    existing = _METRIC_REGISTRY.get(name)
    if existing is not None and existing is not fn:
        raise ConfigError(
            f"Metric {name!r} is already registered by a different plugin"
        )
    _METRIC_REGISTRY[name] = fn
    if lower_is_better:
        _METRIC_LOWER_IS_BETTER.add(name)


def get_metric_fn(name: str) -> MetricFn | None:
    """Return the registered metric function for a name, or None. Loads entry points."""
    ensure_entry_points_loaded()
    return _METRIC_REGISTRY.get(name)


def registered_metric_names() -> list[str]:
    """Return the sorted list of currently registered plugin metric names."""
    ensure_entry_points_loaded()
    return sorted(_METRIC_REGISTRY)


def metric_is_lower_is_better(name: str) -> bool:
    """Whether a registered plugin metric is lower-is-better."""
    ensure_entry_points_loaded()
    return name in _METRIC_LOWER_IS_BETTER


def register_reporter(name: str, fn: ReporterFn) -> None:
    """Register a report sink invoked after a run with a :class:`ReportContext`.

    Raises ``ConfigError`` on an empty name, a non-callable, or a conflicting
    re-registration.
    """
    if not name:
        raise ConfigError("Reporter plugin name must be non-empty")
    if not callable(fn):
        raise ConfigError("Reporter plugin must be a (ReportContext) -> None callable")

    existing = _REPORTER_REGISTRY.get(name)
    if existing is not None and existing is not fn:
        raise ConfigError(
            f"Reporter {name!r} is already registered by a different plugin"
        )
    _REPORTER_REGISTRY[name] = fn


def get_reporter(name: str) -> ReporterFn | None:
    """Return the registered reporter for a name, or None. Loads entry points."""
    ensure_entry_points_loaded()
    return _REPORTER_REGISTRY.get(name)


def registered_reporter_names() -> list[str]:
    """Return the sorted list of currently registered reporter names."""
    ensure_entry_points_loaded()
    return sorted(_REPORTER_REGISTRY)


def load_module_plugins(modules: list[str]) -> None:
    """Import dotted module paths so their top-level ``register_judge`` calls run.

    The current working directory (the config's directory at load time) is placed on
    ``sys.path`` so a local, in-repo plugin module resolves without packaging.
    """
    if not modules:
        return

    import os
    import sys

    cwd = os.getcwd()
    added_cwd = cwd not in sys.path
    if added_cwd:
        sys.path.insert(0, cwd)
    try:
        for module_path in modules:
            try:
                importlib.import_module(module_path)
            except ImportError as e:
                raise ConfigError(
                    f"Failed to import plugin module {module_path!r}: {e}"
                ) from e
    finally:
        if added_cwd:
            try:
                sys.path.remove(cwd)
            except ValueError:
                pass


def ensure_entry_points_loaded() -> None:
    """Discover and register judge/metric plugins advertised via entry-point groups.

    Idempotent: entry points are scanned once per process. A plugin that fails to load
    is skipped with a warning rather than crashing the run.
    """
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    for ep in _select_entry_points(ENTRY_POINT_GROUP):
        try:
            register_judge(ep.name, ep.load())
        except ConfigError:
            # Collision or bad shape — re-raise so misconfiguration is visible.
            raise
        except Exception as e:  # pragma: no cover - defensive against broken plugins
            import warnings

            warnings.warn(
                f"Failed to load llmci judge plugin {ep.name!r}: {e}", stacklevel=2
            )

    for ep in _select_entry_points(ENTRY_POINT_GROUP_METRICS):
        try:
            register_metric(ep.name, ep.load())
        except ConfigError:
            raise
        except Exception as e:  # pragma: no cover - defensive against broken plugins
            import warnings

            warnings.warn(
                f"Failed to load llmci metric plugin {ep.name!r}: {e}", stacklevel=2
            )

    for ep in _select_entry_points(ENTRY_POINT_GROUP_REPORTERS):
        try:
            register_reporter(ep.name, ep.load())
        except ConfigError:
            raise
        except Exception as e:  # pragma: no cover - defensive against broken plugins
            import warnings

            warnings.warn(
                f"Failed to load llmci reporter plugin {ep.name!r}: {e}", stacklevel=2
            )


def _select_entry_points(group: str) -> list:
    """Return entry points in a group across importlib.metadata versions."""
    from importlib.metadata import entry_points

    try:
        # Python 3.10+: selectable interface.
        return list(entry_points(group=group))
    except TypeError:  # pragma: no cover - older API fallback
        return list(entry_points().get(group, []))


def reset_registry() -> None:
    """Clear the registries and entry-point cache. For tests only."""
    global _entry_points_loaded
    _JUDGE_REGISTRY.clear()
    _METRIC_REGISTRY.clear()
    _METRIC_LOWER_IS_BETTER.clear()
    _REPORTER_REGISTRY.clear()
    _entry_points_loaded = False


__all__ = [
    "JudgeFactory", "ENTRY_POINT_GROUP", "ENTRY_POINT_GROUP_METRICS",
    "ENTRY_POINT_GROUP_REPORTERS", "BUILTIN_JUDGE_TYPES", "register_judge",
    "get_judge_factory", "registered_judge_types", "load_module_plugins",
    "ensure_entry_points_loaded", "reset_registry", "MetricContext", "MetricFn",
    "register_metric", "get_metric_fn", "registered_metric_names",
    "metric_is_lower_is_better", "ReportContext", "ReporterFn", "register_reporter",
    "get_reporter", "registered_reporter_names",
]
