r"""Implement an evaluator that does nothing."""

from __future__ import annotations

__all__ = ["NoOpEvaluator"]

from typing import TYPE_CHECKING, Any

from argos.meta_agent.evaluator.base import BaseEvaluator
from argos.meta_agent.typing import InputT, OutputT, TargetT

if TYPE_CHECKING:
    from argos.meta_agent.benchmark import Benchmark
    from argos.meta_agent.prediction import PredictionResult


class NoOpEvaluator(BaseEvaluator[InputT, TargetT, OutputT]):
    r"""Implement an evaluator that does nothing.

    This evaluator should be used if no metrics are desired. It always
    returns an empty dictionary.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def evaluate(
        self,
        predictions: PredictionResult[OutputT],  # noqa: ARG002
        benchmark: Benchmark[InputT, TargetT],  # noqa: ARG002
    ) -> dict[Any, Any]:
        return {}
