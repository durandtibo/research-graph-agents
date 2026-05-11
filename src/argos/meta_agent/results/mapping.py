r"""Implement a result that is a container for a dict of results."""

from __future__ import annotations

__all__ = ["ResultDict"]

from dataclasses import dataclass
from typing import Any

from coola.equality import objects_are_equal
from coola.utils.format import (
    repr_indent,
    repr_mapping,
    str_indent,
    str_mapping,
)

from argos.meta_agent.results.base import BaseResult


@dataclass(frozen=True)
class ResultDict(BaseResult):
    r"""Implement a result that is a container for a dict of results.

    Args:
        results: A mapping from string keys to :class:`BaseResult`
            instances. Each key typically identifies a dataset split or
            evaluation phase (e.g. ``"train"``, ``"val"``).

    Note:
        :meth:`to_markdown` renders one top-level bullet per key and
        nests the child result markdown underneath it. Empty mappings
        return ``"_No metrics available._"``.

    Example:
        ```pycon
        >>> from argos.meta_agent.results import Result, ResultDict
        >>> result = ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
        >>> result.to_dict()
        {'train': {'loss': 0.5}, 'val': {'loss': 0.3}}
        >>> result.to_flat_dict()
        {'train.loss': 0.5, 'val.loss': 0.3}
        >>> print(result.to_markdown())
        - **train**:
          - **loss**: 0.5
        - **val**:
          - **loss**: 0.3

        ```
    """

    results: dict[str, BaseResult]

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self.results))
        if args:
            args = f"\n  {args}\n"
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self.results))
        if args:
            args = f"\n  {args}\n"
        return f"{self.__class__.__qualname__}({args})"

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.results, other.results, equal_nan=equal_nan)

    def to_dict(self) -> dict[str, Any]:
        return {key: value.to_dict() for key, value in self.results.items()}

    def to_markdown(self) -> str:
        if not self.results:
            return "_No metrics available._"
        metrics = [
            f"- **{key}**:\n  {str_indent(value.to_markdown())}"
            for key, value in self.results.items()
        ]
        return "\n".join(metrics)
