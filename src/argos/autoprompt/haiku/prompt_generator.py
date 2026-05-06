r"""Contain prompt generators."""

from __future__ import annotations

__all__ = ["BasePromptGenerator", "HistoryPromptGenerator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping
from iden.io import save_json

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.runnables import Runnable

    from argos.models.prompt_generation import (
        PromptGeneratorInput,
        PromptGeneratorOutput,
    )


class BasePromptGenerator(ABC):
    r"""Abstract base class for prompt generators.

    Subclasses must implement :meth:`generate` to produce a new system
    prompt string, typically by consulting the history of past prompts
    and their associated evaluation metrics.
    """

    @abstractmethod
    def generate(self) -> str:
        r"""Generate a new system prompt.

        Returns:
            The generated system prompt string to be used in the
                next iteration of the optimization loop.
        """


class HistoryPromptGenerator(BasePromptGenerator):
    r"""Generate a new prompt based on the history of previous prompts.

    Args:
        history: The history of previous prompts, as a list of dicts
            where each entry captures the prompt and associated metrics
            from one past iteration.
        model: The :class:`~langchain_core.runnables.Runnable` used to
            generate the next prompt from the history.
        path: An optional path where the raw model output is saved as
            a JSON file. If ``None``, no file is written.
    """

    def __init__(
        self,
        history: list[dict[str, Any]],
        model: Runnable[PromptGeneratorInput, PromptGeneratorOutput],
        path: Path | None = None,
    ) -> None:
        self._history = history
        self._model = model
        self._path = path

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self) -> str:
        history_str = (
            f"The prompt history is provided below as a JSON array. "
            f"Items are listed in order of execution, starting with the "
            f"first iteration and ending with the most recent."
            f"\n{self._history}"
        )
        out = self._model.invoke({"history": history_str})
        if self._path:
            save_json(out.model_dump(), self._path, exist_ok=True)
        return out.prompt

    def _get_kwargs(self) -> dict[str, Any]:
        return {
            "history": self._history,
            "model": self._model,
            "path": self._path,
        }
