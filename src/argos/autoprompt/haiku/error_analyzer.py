r"""Contain abstractions and implementations for error analyzers.

An error analyzer calls an inner error finder to locate mispredicted
examples, then passes the formatted report to an LLM to produce a
textual analysis. The analysis is returned as a string and can
optionally be saved to disk.
"""

from __future__ import annotations

__all__ = ["BaseErrorAnalyzer", "ErrorAnalyzer"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping
from iden.io import save_text
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from langchain_core.runnables import Runnable

    from argos.autoprompt.haiku.error_finder import BaseErrorFinder
    from argos.models.analysis import AnalyzerInput


class BaseErrorAnalyzer(ABC):
    r"""Abstract base class for error analyzers.

    Subclasses must implement :meth:`analyze` to inspect a predictions
    DataFrame, identify mispredicted examples, and return a textual
    analysis suitable for use as LLM context in the next optimization
    iteration.
    """

    @abstractmethod
    def analyze(self, predictions: pl.DataFrame) -> str:
        r"""Compute a textual analysis of the prediction errors.

        Args:
            predictions: A :class:`~polars.DataFrame` produced by the
                haiku judge, containing prediction and target columns.

        Returns:
            A string summarising the error patterns found in
                ``predictions``.
        """


class ErrorAnalyzer(BaseErrorAnalyzer):
    r"""Implement a simple error analyzer.

    Args:
        error_finder: An :class:`~argos.autoprompt.haiku.error_finder.BaseErrorFinder`
            that inspects a predictions DataFrame and returns a
            formatted error report string.
        model: The :class:`~langchain_core.runnables.Runnable` used to
            generate a textual analysis of the errors. It accepts an
            :class:`~argos.models.analysis.AnalyzerInput` dict and
            returns either an :class:`~langchain_core.messages.AIMessage`
            or a dict with an ``"analysis"`` key.
        path: An optional path where the analysis text is saved as a
            plain-text file. If ``None``, no file is written.
    """

    def __init__(
        self,
        error_finder: BaseErrorFinder,
        model: Runnable[AnalyzerInput, AIMessage | dict[str, str]],
        path: Path | None = None,
    ) -> None:
        self._error_finder = error_finder
        self._model = model
        self._path = path

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def analyze(self, predictions: pl.DataFrame) -> str:
        r"""Generate a textual analysis from prediction errors.

        The method first delegates to ``error_finder`` to build a markdown
        report of mispredicted examples. It then invokes ``model`` with
        ``{"text": report}`` and extracts the analysis from either
        ``AIMessage.content`` or the ``"analysis"`` key of a dict output.
        When ``path`` is set, the analysis text is written to disk.

        Args:
            predictions: A :class:`~polars.DataFrame` with predictions and
                targets used to derive errors.

        Returns:
            The generated error analysis text.

        Raises:
            KeyError: If the model returns a dict without an
                ``"analysis"`` key.
        """
        errors = self._error_finder.find(predictions)
        out = self._model.invoke({"text": errors})
        analysis = out.content if isinstance(out, AIMessage) else out["analysis"]
        if self._path:
            save_text(analysis, self._path, exist_ok=True)
        return analysis

    def _get_kwargs(self) -> dict[str, Any]:
        return {
            "error_finder": self._error_finder,
            "model": self._model,
            "path": self._path,
        }
