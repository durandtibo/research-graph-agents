r"""Define the base class to implement an evaluator."""

from __future__ import annotations

__all__ = ["BaseEvaluator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic

from argos.meta_agent.typing import InputT, OutputT, TargetT

if TYPE_CHECKING:
    from argos.meta_agent.prediction import Benchmark, PredictionResult


class BaseEvaluator(ABC, Generic[InputT, TargetT, OutputT]):
    r"""Define the base class to implement an evaluator.

    Subclasses must implement :meth:`evaluate` to compare agent
    predictions against benchmark targets and return a dictionary of
    metrics.

    Example:
        ```pycon
        >>> from argos.meta_agent.benchmark import Benchmark
        >>> from argos.meta_agent.evaluators import BaseEvaluator, NoOpEvaluator
        >>> from argos.meta_agent.prediction import PredictionResult
        >>> evaluator = NoOpEvaluator()
        >>> isinstance(evaluator, BaseEvaluator)
        True

        ```
    """

    @abstractmethod
    def evaluate(
        self,
        predictions: PredictionResult[OutputT],
        benchmark: Benchmark[InputT, TargetT],
    ) -> dict[Any, Any]:
        r"""Evaluate the performance of the given predictions.

        Args:
            predictions: The predictions to evaluate.
            benchmark: The benchmark containing the ground-truth
                targets.

        Returns:
            A dictionary mapping metric names to their computed
                values.

        Example:
            ```pycon
            >>> from argos.meta_agent.benchmark import Benchmark
            >>> from argos.meta_agent.evaluators import NoOpEvaluator
            >>> from argos.meta_agent.prediction import PredictionResult
            >>> evaluator = NoOpEvaluator()
            >>> evaluator.evaluate(PredictionResult(records=[]), Benchmark({}))
            {}

            ```
        """
