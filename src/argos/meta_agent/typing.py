r"""Contain some typing utilities."""

from __future__ import annotations

__all__ = ["InputT", "OutputT", "TargetT"]

from typing import TypeAlias, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
TargetT = TypeVar("TargetT")

FlatDict: TypeAlias = dict[str, int | float | str | bool | None]
r"""A flat dictionary whose values are JSON-compatible scalar types.

Keys are always strings. Values are restricted to ``int``, ``float``,
``str``, ``bool``, or ``None``.
"""
