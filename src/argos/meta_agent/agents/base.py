r"""Contain the base class for all agents."""

from __future__ import annotations

__all__ = ["BaseAgent"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from argos.meta_agent.typing import InputT, OutputT

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class BaseAgent(ABC, Generic[InputT, OutputT]):
    r"""Define the base class for all agents."""

    @abstractmethod
    def predict(self, inputs: list[InputT], config: RunnableConfig | None = None) -> list[OutputT]:
        r"""Make predictions.

        Args:
            inputs: The inputs of the agent.
            config: A config to use when invoking the `Runnable`.

        Returns:
            The predictions of the agent.
        """
