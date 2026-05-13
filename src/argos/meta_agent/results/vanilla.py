r"""Implement a simple result implementation for flat dictionary
metrics."""

from __future__ import annotations

__all__ = ["Result"]

from typing import Any

from attr import dataclass
from coola.equality import objects_are_equal

from argos.meta_agent.results.base import BaseResult


@dataclass(frozen=True)
class Result(BaseResult):
    r"""Define a simple result implementation for flat dictionary
    metrics.

    Args:
        metrics: A flat dict of metrics whose keys are strings and
            values are JSON-compatible scalar types (``int``, ``float``,
            ``str``, ``bool``, or ``None``).

    Note:
        :meth:`to_markdown` renders one Markdown bullet per metric. When
        the result is empty it returns ``"_No metrics available._"``.

    Example:
        ```pycon
        >>> from argos.meta_agent.results import Result
        >>> result = Result({"loss": 0.5, "accuracy": 0.9})
        >>> result
        Result(metrics={'loss': 0.5, 'accuracy': 0.9})
        >>> result.to_dict()
        {'loss': 0.5, 'accuracy': 0.9}
        >>> print(result.to_markdown())
        - loss: 0.5
        - accuracy: 0.9

        ```
    """

    metrics: dict[str, Any]

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.metrics, other.metrics, equal_nan=equal_nan)

    def to_dict(self) -> dict[str, Any]:
        return self.metrics

    def to_markdown(self) -> str:
        if not self.metrics:
            return "_No metrics available._"
        metrics = [f"- {key}: {value}" for key, value in self.metrics.items()]
        return "\n".join(metrics)
