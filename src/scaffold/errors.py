"""Custom exception types for Scaffold."""

from __future__ import annotations


class ScaffoldError(Exception):
    """Base exception for all Scaffold errors."""

    pass


class ConfigError(ScaffoldError):
    """Error in scaffold.yaml configuration."""

    pass


class DatasetError(ScaffoldError):
    """Error loading or validating an eval dataset."""

    pass


class TargetError(ScaffoldError):
    """Error executing a target."""

    pass
