r"""Contain some agent factory classes that are used to create agents
based on the configuration optimized by the meta-agent."""

from __future__ import annotations

__all__ = ["BaseAgentFactory"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from argos.meta_agent.agents.base import BaseAgent
    from argos.meta_agent.agents.config import AgentConfig


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseAgentFactory(ABC, Generic[InputT, OutputT]):
    r"""Define the base class to implement an agent factory.

    Subclasses must implement :meth:`create` to instantiate a
    :class:`~argos.meta_agent.agents.BaseAgent` from an
    :class:`~argos.meta_agent.agents.AgentConfig`.

    Example:
        ```pycon
        >>> from langchain_core.runnables import RunnableLambda
        >>> from argos.meta_agent.agents import (
        ...     Agent,
        ...     AgentConfig,
        ...     BaseAgent,
        ...     BaseAgentFactory,
        ... )
        >>> class UpperCaseAgentFactory(BaseAgentFactory):
        ...     def create(self, config: AgentConfig) -> BaseAgent:
        ...         return Agent(RunnableLambda(str.upper))
        ...
        >>> factory = UpperCaseAgentFactory()
        >>> agent = factory.create(AgentConfig(components={}))
        >>> agent.predict(["hello", "world"])
        ['HELLO', 'WORLD']

        ```
    """

    @abstractmethod
    def create(self, config: AgentConfig) -> BaseAgent[InputT, OutputT]:
        r"""Instantiate an agent from its configuration.

        Args:
            config: The configuration of the agent that is
                optimized by the meta-agent.

        Returns:
            A new agent instance built from the given
                configuration.

        Example:
            ```pycon
            >>> from langchain_core.runnables import RunnableLambda
            >>> from argos.meta_agent.agents import (
            ...     Agent,
            ...     AgentConfig,
            ...     BaseAgent,
            ...     BaseAgentFactory,
            ... )
            >>> class UpperCaseAgentFactory(BaseAgentFactory):
            ...     def create(self, config: AgentConfig) -> BaseAgent:
            ...         return Agent(RunnableLambda(str.upper))
            ...
            >>> factory = UpperCaseAgentFactory()
            >>> agent = factory.create(AgentConfig(components={}))
            >>> agent.predict(["hello"])
            ['HELLO']

            ```
        """
