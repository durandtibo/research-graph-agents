r"""Contain some agent factory classes that are used to create agents
based on the configuration optimized by the meta-agent."""

from __future__ import annotations

__all__ = ["BaseAgentFactory"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argos.meta_agent.agent.base import BaseAgent
    from argos.meta_agent.agent.config import AgentConfig


class BaseAgentFactory(ABC):
    r"""Define the base class to implement an agent factory."""

    @abstractmethod
    def create(self, config: AgentConfig) -> BaseAgent:
        r"""Instantiate an agent from its configuration.

        Args:
            config: The configuration of the agent that is optimized by the meta-agent.
        """
