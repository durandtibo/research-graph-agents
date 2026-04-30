r"""Contain some typing utilities."""

from __future__ import annotations

__all__ = ["InputT", "OutputT", "TargetT"]

from typing import TypeAlias, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
TargetT = TypeVar("TargetT")

FlatDict: TypeAlias = dict[str, int | float | str | bool | None]
