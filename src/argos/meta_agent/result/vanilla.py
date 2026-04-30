r"""Implement a simple result implementation for flat dictionary
metrics."""

from __future__ import annotations

__all__ = ["Result"]

import copy
from typing import TYPE_CHECKING

from coola.equality import objects_are_equal
from coola.nested import to_flat_dict
from coola.utils.format import repr_mapping_line, str_mapping_line

from argos.meta_agent.result.base import BaseResult

if TYPE_CHECKING:
    from argos.meta_agent.typing import FlatDict


class Result(BaseResult):
    r"""Define a simple result implementation for flat dictionary
    metrics.

    Args:
        metrics: A flat dict of metrics.
    """

    def __init__(self, metrics: FlatDict) -> None:
        self._metrics = metrics

    def __repr__(self) -> str:
        args = repr_mapping_line(self._metrics)
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = str_mapping_line(self._metrics)
        return f"{self.__class__.__qualname__}({args})"

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self._metrics, other._metrics, equal_nan=equal_nan)

    def to_dict(self) -> FlatDict:
        return self.to_raw_dict()

    def to_flat_dict(self, separator: str = ".") -> FlatDict:
        return to_flat_dict(self._metrics, separator=separator)

    def to_raw_dict(self) -> FlatDict:
        return copy.copy(self._metrics)

    def to_markdown(self) -> str:
        if not self._metrics:
            return "_No metrics available._"
        metrics = [f"- **{key}**: {value}" for key, value in self._metrics.items()]
        return "\n".join(metrics)
