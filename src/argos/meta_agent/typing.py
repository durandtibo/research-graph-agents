r"""Contain some typing utilities."""

from __future__ import annotations

__all__ = ["InputT", "OutputT", "TargetT"]

from typing import TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
TargetT = TypeVar("TargetT")
