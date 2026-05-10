r"""Define the base class to implement an evaluator."""

from __future__ import annotations

__all__ = ["BaseEvaluator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from argos.meta_agent.results import BaseResult


class BaseEvaluator(ABC):
    r"""Define the base class to implement an evaluator.

    Subclasses must implement :meth:`evaluate` to compare agent
    predictions against benchmark targets and return a
    :class:`~argos.meta_agent.results.BaseResult` object.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.meta_agent.evaluators import NoOpEvaluator
        >>> evaluator = NoOpEvaluator()
        >>> data = pl.DataFrame({"id": ["q1", "q2", "q3"]})
        >>> result = evaluator.evaluate(data)
        >>> result
        Result()

        ```
    """

    @abstractmethod
    def evaluate(self, data: pl.DataFrame) -> BaseResult:
        r"""Evaluate the performance of the given data.

        Args:
            data: The data used to evaluate the performance.

        Returns:
            A result object containing the computed evaluation
                metrics.

        Example:
            ```pycon
            >>> import polars as pl
            >>> from argos.meta_agent.evaluators import NoOpEvaluator
            >>> evaluator = NoOpEvaluator()
            >>> data = pl.DataFrame({"id": ["q1", "q2", "q3"]})
            >>> result = evaluator.evaluate(data)
            >>> result
            Result()

            ```
        """

    @abstractmethod
    def equal(self, other: object, equal_nan: bool = False) -> bool:
        r"""Return ``True`` if the two objects are equal, otherwise
        ``False``.

        Args:
            other: The value to compare with.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``
        """
