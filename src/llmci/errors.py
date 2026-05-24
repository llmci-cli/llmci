"""Custom exception types for llmci."""

from __future__ import annotations


class LlmciError(Exception):
    """Base exception for all llmci errors."""

    pass


class ConfigError(LlmciError):
    """Error in llmci.yaml configuration."""

    pass


class DatasetError(LlmciError):
    """Error loading or validating an eval dataset."""

    pass


class TargetError(LlmciError):
    """Error executing a target."""

    pass
