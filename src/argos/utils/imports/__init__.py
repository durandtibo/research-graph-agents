r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_colorlog",
    "colorlog_available",
    "is_colorlog_available",
    "raise_colorlog_missing_error",
]

from argos.utils.imports.colorlog import (
    check_colorlog,
    colorlog_available,
    is_colorlog_available,
    raise_colorlog_missing_error,
)
