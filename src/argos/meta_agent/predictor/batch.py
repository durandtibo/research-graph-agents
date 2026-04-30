r"""Define the base class to implement a predictor."""

from __future__ import annotations

__all__ = ["BatchPredictor"]

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping
from coola.utils.timing import timeblock
from langchain_core.runnables import RunnableConfig

from argos.meta_agent.interface import PredictionResult
from argos.meta_agent.predictor.base import BasePredictor
from argos.utils.batching import batchify

if TYPE_CHECKING:
    from argos.meta_agent.agent import BaseAgent
    from argos.meta_agent.interface import Benchmark


logger: logging.Logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
PredictionT = TypeVar("PredictionT")
TargetT = TypeVar("TargetT")


class BatchPredictor(BasePredictor[InputT, TargetT, PredictionT]):
    r"""Define a predictor that computes predictions by batches."""

    def __init__(
        self,
        batch_size: int = 1,
        config: RunnableConfig | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._config = config or RunnableConfig(max_concurrency=batch_size)

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

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
        batches = batchify(list(benchmark.examples.values()), size=self._batch_size)
        predictions = []
        with timeblock(message="LLM inference time: {time}"):
            for index, batch in enumerate(batches):
                logger.info(f"--- Processing Batch {index + 1} ---")
                predictions.extend(
                    agent.predict(inputs=[example.input for example in batch], config=self._config)
                )

        return PredictionResult.from_predictions(
            example_ids=list(benchmark.examples.keys()),
            predictions=predictions,
        )

    def _get_kwargs(self) -> dict[str, Any]:
        return {"batch_size": self._batch_size, "config": self._config}
