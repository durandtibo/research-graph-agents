r"""Implement an evaluator that does nothing."""

from __future__ import annotations

__all__ = ["NoOpEvaluator"]

from typing import TYPE_CHECKING

from argos.meta_agent.evaluators.base import BaseEvaluator
from argos.meta_agent.results import Result

if TYPE_CHECKING:
    import polars as pl


class NoOpEvaluator(BaseEvaluator):
    r"""Implement an evaluator that does nothing.

    This evaluator should be used if no metrics are desired. It always
    returns an empty result.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.meta_agent.evaluators import NoOpEvaluator
        >>> evaluator = NoOpEvaluator()
        >>> data = pl.DataFrame({"id": ["q1", "q2", "q3"]})
        >>> result = evaluator.evaluate(data)
        >>> result
        Result(metrics={})

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def evaluate(self, data: pl.DataFrame) -> Result:  # noqa: ARG002
        return Result({})

    def equal(self, other: object, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return type(other) is type(self)
