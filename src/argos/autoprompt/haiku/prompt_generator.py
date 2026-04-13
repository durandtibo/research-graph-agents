r"""Contain prompt generators."""

from __future__ import annotations

__all__ = ["BasePromptGenerator", "HistoryPromptGenerator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola.utils.format import repr_indent, repr_mapping

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import Runnable

    from argos.models.analysis import AnalyzerInput


class BasePromptGenerator(ABC):
    r"""Define the base class to implement a prompt generator."""

    @abstractmethod
    def generate(self) -> str:
        r"""Generate a prompt.

        Returns:
            The generated prompt.
        """


class HistoryPromptGenerator(BasePromptGenerator):
    r"""Generate a new prompt based on the history of previous
    prompts."""

    def __init__(
        self,
        history: list[dict[str, Any]],
        model: Runnable[AnalyzerInput, AIMessage],
    ) -> None:
        self._history = history
        self._model = model

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"history": self._history, "model": self._model}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self) -> str:
        history_str = (
            f"The prompt history is provided below as a JSON array. "
            f"Items are listed in order of execution, starting with the "
            f"first iteration and ending with the most recent. "
            f"\n{self._history}"
        )
        out = self._model.invoke({"text": history_str})
        return out.content
