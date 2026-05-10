r"""Contain type variables and type aliases used across the meta-agent.

``InputT``, ``OutputT``, and ``TargetT`` are generic type variables that
parameterize agents, predictors, datasets, and examples. ``FlatDict`` is
a type alias for flat, JSON-compatible dictionaries.
"""

from __future__ import annotations

__all__ = ["InputT", "OutputT", "TargetT"]

from typing import TypeAlias, TypeVar

InputT = TypeVar("InputT")
r"""Type variable for the agent's input type."""

OutputT = TypeVar("OutputT")
r"""Type variable for the agent's output type."""

TargetT = TypeVar("TargetT")
r"""Type variable for the benchmark's target type."""

FlatDict: TypeAlias = dict[str, int | float | str | bool | None]
r"""A flat dictionary whose values are JSON-compatible scalar types.

Keys are always strings. Values are restricted to ``int``, ``float``,
``str``, ``bool``, or ``None``.
"""
