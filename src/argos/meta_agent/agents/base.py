r"""Contain the base class for all agents."""

from __future__ import annotations

__all__ = ["BaseAgent"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from argos.meta_agent.typing import InputT, OutputT

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class BaseAgent(ABC, Generic[InputT, OutputT]):
    r"""Define the base class for all agents.

    Subclasses must implement :meth:`predict` to process a batch of
    inputs and return the corresponding outputs.

    Example:
        ```pycon
        >>> from langchain_core.runnables import RunnableLambda
        >>> from argos.meta_agent.agents import Agent, BaseAgent
        >>> agent = Agent(RunnableLambda(str.upper))
        >>> isinstance(agent, BaseAgent)
        True
        >>> agent.predict(["hello", "world"])
        ['HELLO', 'WORLD']

        ```
    """

    @abstractmethod
    def predict(self, inputs: list[InputT], config: RunnableConfig | None = None) -> list[OutputT]:
        r"""Make predictions.

        Args:
            inputs: The inputs of the agent.
            config: A config to use when invoking the `Runnable`.

        Returns:
            The predictions of the agent.
        """
