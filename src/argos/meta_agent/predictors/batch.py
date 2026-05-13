r"""Implement a predictor that processes examples in batches."""

from __future__ import annotations

__all__ = ["BatchPredictor"]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils.batching import batchify
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping
from coola.utils.timing import timeblock
from langchain_core.runnables import RunnableConfig

from argos.meta_agent.batches import BaseBatch, Batch
from argos.meta_agent.entities import Prediction
from argos.meta_agent.predictors.base import BasePredictor
from argos.meta_agent.typing import InputT, OutputT

if TYPE_CHECKING:
    from argos.meta_agent.agents import BaseAgent
    from argos.meta_agent.entities import BaseExample, BasePrediction

logger: logging.Logger = logging.getLogger(__name__)


class BatchPredictor(BasePredictor[InputT, OutputT]):
    r"""Define a predictor that computes predictions by batches.

    Args:
        batch_size: Number of examples to process concurrently per
            batch. Defaults to ``1``.
        config: Optional :class:`~langchain_core.runnables.RunnableConfig`
            controlling concurrency. If ``None``, defaults to
            ``RunnableConfig(max_concurrency=batch_size)``.

    Example:
        ```pycon
        >>> from langchain_core.runnables import RunnableLambda
        >>> from argos.meta_agent.agents import Agent
        >>> from argos.meta_agent.benchmark import Benchmark, BenchmarkExample
        >>> from argos.meta_agent.predictors import BatchPredictor
        >>> agent = Agent(RunnableLambda(str.upper))
        >>> benchmark = Benchmark.from_examples(
        ...     [
        ...         BenchmarkExample(id="q1", input="hello", target="HELLO"),
        ...         BenchmarkExample(id="q2", input="world", target="WORLD"),
        ...     ]
        ... )
        >>> predictor = BatchPredictor(batch_size=2)
        >>> result = predictor.predict(agent, benchmark)
        >>> result.to_dict()
        {'q1': 'HELLO', 'q2': 'WORLD'}

        ```
    """

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
        agent: BaseAgent[InputT, OutputT],
        dataset: BaseBatch[BaseExample[InputT]],
    ) -> BaseBatch[BasePrediction[OutputT]]:
        batches = batchify(list(dataset.items.values()), size=self._batch_size)
        predictions = []
        with timeblock(message="LLM inference time: {time}"):
            for index, batch in enumerate(batches):
                logger.info(f"--- Processing Batch {index + 1} ---")
                inputs = [example.input for example in batch]
                outputs = agent.predict(inputs=inputs, config=self._config)
                predictions.extend(
                    [Prediction(id=ex.id, prediction=out) for ex, out in zip(batch, outputs)]
                )

        return Batch.from_list(predictions)

    def _get_kwargs(self) -> dict[str, Any]:
        return {"batch_size": self._batch_size, "config": self._config}
