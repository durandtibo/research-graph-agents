r"""Define the base class to implement a predictor."""

from __future__ import annotations

__all__ = ["BasePredictor"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from argos.meta_agent.typing import InputT, OutputT, TargetT

if TYPE_CHECKING:
    from argos.meta_agent.agents import BaseAgent
    from argos.meta_agent.datasets import BaseDataset
    from argos.meta_agent.predictions import BasePrediction


class BasePredictor(ABC, Generic[InputT, TargetT, OutputT]):
    r"""Define the base class to implement a predictor.

    Subclasses must implement :meth:`predict` to run an agent over all
    examples in a dataset and return the collected
    :class:`~argos.meta_agent.prediction.PredictionResult`.

    Example:
        ```pycon
        >>> from argos.meta_agent.predictors import BatchPredictor
        >>> predictor = BatchPredictor(batch_size=4)
        >>> predictor
        BatchPredictor(
          (batch_size): 4
          (config): {'max_concurrency': 4}
        )

        ```
    """

    @abstractmethod
    def predict(
        self,
        agent: BaseAgent[InputT, OutputT],
        dataset: BaseDataset[InputT, TargetT],
    ) -> BasePrediction[OutputT]:
        r"""Make the prediction result for the input dataset.

        Args:
            agent: The agent used to make predictions.
            dataset: The dataset to run predictions against.

        Returns:
            The prediction result containing one prediction per
                dataset example.

        Example:
            ```pycon
            >>> from langchain_core.runnables import RunnableLambda
            >>> from argos.meta_agent.agents import Agent
            >>> from argos.meta_agent.datasets import Dataset
            >>> from argos.meta_agent.examples import Example
            >>> from argos.meta_agent.predictors import BatchPredictor
            >>> agent = Agent(RunnableLambda(str.upper))
            >>> dataset = Dataset.from_examples([Example(id="q1", input="hello", target="HELLO")])
            >>> predictor = BatchPredictor()
            >>> predictor.predict(agent, dataset).to_dict()
            {'q1': 'HELLO'}

            ```
        """
