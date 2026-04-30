r"""Define the base class to implement a predictor."""

from __future__ import annotations

__all__ = ["BasePredictor"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from argos.meta_agent.agent import BaseAgent
    from argos.meta_agent.interface import Benchmark, PredictionResult

InputT = TypeVar("InputT")
PredictionT = TypeVar("PredictionT")
TargetT = TypeVar("TargetT")


class BasePredictor(ABC, Generic[InputT, TargetT, PredictionT]):
    r"""Define the base class to implement a predictor."""

    @abstractmethod
    def predict(
        self,
        agent: BaseAgent[InputT, PredictionT],
        benchmark: Benchmark[InputT, TargetT],
    ) -> PredictionResult[PredictionT]:
        r"""Make the prediction result for the input benchmark.

        Args:
            agent: The agent to make prediction for.
            benchmark: The benchmark to make prediction for.

        Returns:
            The prediction result.
        """
