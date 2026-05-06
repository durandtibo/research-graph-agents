r"""Implement a simple result implementation for flat dictionary
metrics."""

from __future__ import annotations

__all__ = ["Result"]

from typing import TYPE_CHECKING

from coola.equality import objects_are_equal
from coola.nested import to_flat_dict
from coola.utils.format import repr_mapping_line, str_mapping_line

from argos.meta_agent.results.base import BaseResult

if TYPE_CHECKING:
    from argos.meta_agent.typing import FlatDict


class Result(BaseResult):
    r"""Define a simple result implementation for flat dictionary
    metrics.

    Args:
        metrics: A flat dict of metrics whose keys are strings and
            values are JSON-compatible scalar types (``int``, ``float``,
            ``str``, ``bool``, or ``None``).

    Example:
        ```pycon
        >>> from argos.meta_agent.results import Result
        >>> result = Result({"loss": 0.5, "accuracy": 0.9})
        >>> result.to_dict()
        {'loss': 0.5, 'accuracy': 0.9}
        >>> result.to_markdown()
        '- **loss**: 0.5\n- **accuracy**: 0.9'

        ```
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
            >>> from argos.meta_agent.results import Result
            >>> result = Result({"loss": 0.5, "accuracy": 0.9})
            >>> result.to_flat_dict()
            {'loss': 0.5, 'accuracy': 0.9}

            ```
        """
        return to_flat_dict(self.to_dict(), separator=separator)

    def to_raw_dict(self) -> FlatDict:
        return self._metrics

    def to_markdown(self) -> str:
        r"""Return the result formatted as a Markdown bullet list.

        Each metric is rendered as ``- **<key>**: <value>``.
        Returns ``"_No metrics available._"`` when the result is empty.

        Returns:
            A Markdown string with one bullet per metric, or
                ``"_No metrics available._"`` when the metrics
                dictionary is empty.

        Example:
            ```pycon
            >>> from argos.meta_agent.results import Result
            >>> result = Result({"loss": 0.5, "accuracy": 0.9})
            >>> print(result.to_markdown())
            - **loss**: 0.5
            - **accuracy**: 0.9

            ```
        """
        if not self._metrics:
            return "_No metrics available._"
        metrics = [f"- **{key}**: {value}" for key, value in self._metrics.items()]
        return "\n".join(metrics)
