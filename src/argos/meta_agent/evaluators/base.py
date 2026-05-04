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
    predictions against benchmark targets and return a dictionary of
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
    def evaluate(self, data: pl.DataFrame) -> BaseResult:
        r"""Evaluate the performance of the given data.

        Args:
            data: The data used to evaluate the performance.

        Returns:
            A dictionary mapping metric names to their computed
                values.

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
