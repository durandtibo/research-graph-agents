from __future__ import annotations

__all__ = ["BaseErrorAnalyzer", "ErrorAnalyzer"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from coola.utils.format import repr_indent, repr_mapping
from iden.io import save_text

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from langchain_core.runnables import Runnable

    from argos.autoprompt.haiku.error_finder import BaseErrorFinder


class BaseErrorAnalyzer(ABC):
    r"""Define the base class to implement an error analyzer."""

    @abstractmethod
    def analyze(self, predictions: pl.DataFrame) -> str:
        r"""Compute a textual analysis of the errors.

        Args:
            predictions: A DataFrame with the predictions.

        Returns:
            The analysis of the errors.
        """


class ErrorAnalyzer(BaseErrorAnalyzer):
    r"""Implement a simple error analyzer.

    Args:
        error_finder: An error finder.
        model: The model to generate the analysis of the errors.
        path: An optional path where the analysis will be saved.
    """

    def __init__(
        self,
        error_finder: BaseErrorFinder,
        model: Runnable[dict[str, str], str],
        path: Path | None = None,
    ) -> None:
        self._error_finder = error_finder
        self._model = model
        self._path = path

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "error_finder": self._error_finder,
                    "model": self._model,
                    "path": self._path,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def analyze(self, predictions: pl.DataFrame) -> str:
        errors = self._error_finder.find(predictions)
        analysis = self._model.invoke({"text": errors})
        if self._path:
            save_text(analysis, self._path, exist_ok=True)
        return analysis
