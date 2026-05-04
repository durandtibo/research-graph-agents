r"""Contain abstractions and implementations for agents.

This sub-package provides abstract base classes for agents and
agent factories, along with a concrete runnable-based agent
implementation and a dataclass for agent configuration.
"""

__all__ = ["Agent", "AgentConfig", "BaseAgent", "BaseAgentFactory"]

from argos.meta_agent.agents.base import BaseAgent
from argos.meta_agent.agents.config import AgentConfig
from argos.meta_agent.agents.factory import BaseAgentFactory
from argos.meta_agent.agents.vanilla import Agent
