r"""Implement a simple agent."""

from __future__ import annotations

__all__ = ["Agent"]


from typing import TYPE_CHECKING, TypeVar

from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

from argos.meta_agent.agents.base import BaseAgent

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable, RunnableConfig

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Agent(BaseAgent[InputT, OutputT]):
    r"""Define a simple agent that is a wrapper around a runnable object.

    Args:
        runnable: An instance of a runnable object.

    Example:
        ```pycon
        >>> from langchain_core.runnables import RunnableLambda
        >>> from argos.meta_agent.agents import Agent
        >>> agent = Agent(RunnableLambda(str.upper))
        >>> agent
        Agent(
          (runnable): RunnableLambda(upper)
        )
        >>> agent.predict(["hello", "world"])
        ['HELLO', 'WORLD']

        ```
    """

    def __init__(self, runnable: Runnable[InputT, OutputT]) -> None:
        self._runnable = runnable

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"runnable": self._runnable}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"runnable": self._runnable}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def predict(self, inputs: list[InputT], config: RunnableConfig | None = None) -> list[OutputT]:
        return self._runnable.batch(inputs=inputs, config=config)
