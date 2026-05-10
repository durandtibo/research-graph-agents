r"""Implement a result that is a container for a dict of results."""

from __future__ import annotations

__all__ = ["ResultDict"]

from typing import TYPE_CHECKING, Any

from coola.equality import objects_are_equal
from coola.nested import to_flat_dict
from coola.utils.format import (
    str_indent,
    str_mapping,
)

from argos.meta_agent.results.base import BaseResult

if TYPE_CHECKING:
    from collections.abc import Mapping


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

    def __init__(self, results: Mapping[str, BaseResult]) -> None:
        self._results = dict(results)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(count={len(self._results):,})"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._results))
        if args:
            args = f"\n  {args}\n"
        return f"{self.__class__.__qualname__}({args})"

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self._results, other._results, equal_nan=equal_nan)

    def to_dict(self) -> dict[str, Any]:
        return {key: value.to_dict() for key, value in self._results.items()}

    def to_flat_dict(self, separator: str = ".") -> dict[str, Any]:
        r"""Return the result as a flat dictionary of native Python
        types.

        Args:
            separator: The separator used to join nested keys when
                flattening. Defaults to ``"."``.

        Returns:
            A flat dictionary mapping metric names to scalar native
                Python values, with no nested dicts or lists.

        Example:
            ```pycon
            >>> from argos.meta_agent.results import Result, ResultDict
            >>> result = ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
            >>> result.to_flat_dict()
            {'train.loss': 0.5, 'val.loss': 0.3}

            ```
        """
        return to_flat_dict(self.to_dict(), separator=separator)

    def to_raw_dict(self) -> dict[str, Any]:
        r"""Return the raw mapping of result objects without conversion.

        Unlike :meth:`to_dict`, the values are returned as-is — each
        value is a :class:`~argos.meta_agent.results.BaseResult`
        instance rather than a plain Python dict.

        Returns:
            A dictionary mapping split names to their raw
                :class:`~argos.meta_agent.results.BaseResult`
                instances.

        Example:
            ```pycon
            >>> from argos.meta_agent.results import Result, ResultDict
            >>> result = ResultDict({"train": Result({"loss": 0.5})})
            >>> raw = result.to_raw_dict()
            >>> type(raw["train"])
            <class 'argos.meta_agent.results.vanilla.Result'>

            ```
        """
        return self._results

    def to_markdown(self) -> str:
        r"""Return the result formatted as a nested Markdown string.

        Each top-level split is rendered as a bold label followed by
        its sub-result's markdown representation, indented by two
        spaces. Returns ``"_No metrics available._"`` when the mapping
        is empty.

        Returns:
            A Markdown string with one section per split, or
                ``"_No metrics available._"`` when the mapping is
                empty.

        Example:
            ```pycon
            >>> from argos.meta_agent.results import Result, ResultDict
            >>> result = ResultDict({
            ...     "train": Result({"loss": 0.5}),
            ...     "val": Result({"loss": 0.3}),
            ... })
            >>> print(result.to_markdown())
            - **train**:
              - **loss**: 0.5
            - **val**:
              - **loss**: 0.3

            ```
        """
        if not self._results:
            return "_No metrics available._"
        metrics = [
            f"- **{key}**:\n  {str_indent(value.to_markdown())}"
            for key, value in self._results.items()
        ]
        return "\n".join(metrics)
