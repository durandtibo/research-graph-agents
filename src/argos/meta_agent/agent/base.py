r"""Contain the base class for all agents."""

from __future__ import annotations

__all__ = ["BaseAgent"]

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    r"""Define the base class for all agents."""

    @abstractmethod
    def predict(self, inputs: list[dict[Any, Any]]) -> list[Any]:
        r"""Make predictions.

        Args:
            inputs: The inputs of the agent.

        Returns:
            The predictions of the agent.
        """
