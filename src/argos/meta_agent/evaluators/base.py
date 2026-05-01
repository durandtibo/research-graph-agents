r"""Define the base class to implement an evaluator."""

from __future__ import annotations

__all__ = ["BaseEvaluator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic

from argos.meta_agent.typing import InputT, OutputT, TargetT

if TYPE_CHECKING:
    from argos.meta_agent.prediction import Benchmark, PredictionResult


class BaseEvaluator(ABC, Generic[InputT, TargetT, OutputT]):
    r"""Define the base class to implement an evaluator."""

    @abstractmethod
    def evaluate(
        self,
        predictions: PredictionResult[OutputT],
        benchmark: Benchmark[InputT, TargetT],
    ) -> dict[Any, Any]:
        r"""Evaluate the performance of the given predictions.

        Args:
            predictions: The predictions to evaluate.
            benchmark: The benchmark with the targets to evaluate the performance.

        Returns:
            The performance result.
        """
