r"""Implement a simple agent."""

from __future__ import annotations

__all__ = ["Agent"]


from typing import TYPE_CHECKING, TypeVar

from argos.meta_agent.agent.base import BaseAgent

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable, RunnableConfig

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Agent(BaseAgent[InputT, OutputT]):
    r"""Define a simple agent that is a wrapper around a runnable object.

    Args:
        model: An instance of a runnable object used as model for the agent.
    """

    def __init__(self, model: Runnable[InputT, OutputT]) -> None:
        self._model = model

    def predict(self, inputs: list[InputT], config: RunnableConfig | None = None) -> list[OutputT]:
        return self._model.batch(inputs=inputs, config=config)
