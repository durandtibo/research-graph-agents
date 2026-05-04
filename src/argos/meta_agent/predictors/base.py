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
    r"""Define the base class to implement a predictor.

    Subclasses must implement :meth:`predict` to run an agent over all
    examples in a benchmark and return the collected
    :class:`~argos.meta_agent.prediction.PredictionResult`.

    Example:
        ```pycon
        >>> from langchain_core.runnables import RunnableLambda
        >>> from argos.meta_agent.agents import Agent
        >>> from argos.meta_agent.benchmark import Benchmark, BenchmarkExample
        >>> from argos.meta_agent.predictors import BasePredictor, BatchPredictor
        >>> predictor = BatchPredictor(batch_size=4)
        >>> isinstance(predictor, BasePredictor)
        True

        ```
    """

    @abstractmethod
    def predict(
        self,
        agent: BaseAgent[InputT, OutputT],
        benchmark: Benchmark[InputT, TargetT],
    ) -> PredictionResult[OutputT]:
        r"""Make the prediction result for the input benchmark.

        Args:
            agent: The agent used to make predictions.
            benchmark: The benchmark to run predictions against.

        Returns:
            The prediction result containing one prediction per
                benchmark example.

        Example:
            ```pycon
            >>> from langchain_core.runnables import RunnableLambda
            >>> from argos.meta_agent.agents import Agent
            >>> from argos.meta_agent.benchmark import (
            ...     Benchmark,
            ...     BenchmarkExample,
            ... )
            >>> from argos.meta_agent.predictors import BatchPredictor
            >>> agent = Agent(RunnableLambda(str.upper))
            >>> benchmark = Benchmark.from_examples(
            ...     [BenchmarkExample(id="q1", input="hello", target="HELLO")]
            ... )
            >>> predictor = BatchPredictor()
            >>> predictor.predict(agent, benchmark).to_dict()
            {'q1': 'HELLO'}

            ```
        """
