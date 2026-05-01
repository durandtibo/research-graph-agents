r"""Define the base class to implement a predictor."""

from __future__ import annotations

__all__ = ["BasePredictor"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from argos.meta_agent.typing import InputT, OutputT, TargetT

if TYPE_CHECKING:
    from argos.meta_agent.agents import BaseAgent
    from argos.meta_agent.prediction import Benchmark, PredictionResult


class BasePredictor(ABC, Generic[InputT, TargetT, OutputT]):
    r"""Define the base class to implement a predictor."""

    @abstractmethod
    def predict(
        self,
        agent: BaseAgent[InputT, OutputT],
        benchmark: Benchmark[InputT, TargetT],
    ) -> PredictionResult[OutputT]:
        r"""Make the prediction result for the input benchmark.

        Args:
            agent: The agent to make prediction for.
            benchmark: The benchmark to make prediction for.

        Returns:
            The prediction result.
        """
