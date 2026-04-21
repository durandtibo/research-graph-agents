r"""Contain the base class for all agents."""

from __future__ import annotations

__all__ = ["BaseAgent"]

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseAgent(ABC, Generic[InputT, OutputT]):
    r"""Define the base class for all agents."""

    @abstractmethod
    def predict(self, inputs: list[InputT]) -> list[OutputT]:
        r"""Make predictions.

        Args:
            inputs: The inputs of the agent.

        Returns:
            The predictions of the agent.
        """
