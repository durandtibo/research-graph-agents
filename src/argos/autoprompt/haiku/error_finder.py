r"""Contain code to find the errors."""

from __future__ import annotations

__all__ = ["BaseErrorFinder", "ErrorFinder"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from coola.utils.format import repr_indent, repr_mapping

from argos.autoprompt.haiku import columns
from argos.autoprompt.haiku.error_analysis import (
    find_structure_errors,
    find_topic_errors,
    format_errors_as_markdown,
)
from argos.utils.logging import log_markdown

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl


class BaseErrorFinder(ABC):
    r"""Define the base class to implement an error analyzer."""

    @abstractmethod
    def find(self, predictions: pl.DataFrame) -> str:
        r"""Compute a textual analysis of the errors.

        Args:
            predictions: A DataFrame with the predictions.

        Returns:
            The analysis of the errors.
        """


class ErrorFinder(BaseErrorFinder):
    r"""Implement a simple error finder.

    Args:
        path: The path where to store the errors.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        haiku_col: str = columns.HAIKU,
        topic_col: str = columns.TOPIC,
        overall_prediction_col: str = columns.OVERALL_PREDICTION,
        overall_target_col: str = columns.OVERALL_TARGET,
        structure_prediction_col: str = columns.STRUCTURE_PREDICTION,
        structure_target_col: str = columns.STRUCTURE_TARGET,
        topic_prediction_col: str = columns.TOPIC_PREDICTION,
        topic_target_col: str = columns.TOPIC_TARGET,
    ) -> None:
        self._path = path
        self._haiku_col = haiku_col
        self._topic_col = topic_col
        self._overall_prediction_col = overall_prediction_col
        self._overall_target_col = overall_target_col
        self._structure_prediction_col = structure_prediction_col
        self._structure_target_col = structure_target_col
        self._topic_prediction_col = topic_prediction_col
        self._topic_target_col = topic_target_col

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "path": self._path,
                    "haiku_col": self._haiku_col,
                    "topic_col": self._topic_col,
                    "overall_prediction_col": self._overall_prediction_col,
                    "overall_target_col": self._overall_target_col,
                    "structure_prediction_col": self._structure_prediction_col,
                    "structure_target_col": self._structure_target_col,
                    "topic_prediction_col": self._topic_prediction_col,
                    "topic_target_col": self._topic_target_col,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def structure_error_path(self) -> Path | None:
        if not self._path:
            return None
        return self._path.joinpath("errors_structure.json")

    @property
    def topic_error_path(self) -> Path | None:
        if not self._path:
            return None
        return self._path.joinpath("errors_topic.json")

    def find(self, predictions: pl.DataFrame) -> str:
        structure = self._find_structure_errors(predictions)
        topic = self._find_topic_errors(predictions)
        return (
            f"## Examples with Structure Errors\n\n{structure}\n\n"
            f"## Examples with Topic Errors\n\n{topic}"
        )

    def _find_structure_errors(self, predictions: pl.DataFrame) -> str:
        errors = find_structure_errors(
            predictions=predictions,
            path=self.structure_error_path,
            haiku_col=self._haiku_col,
            topic_col=self._topic_col,
            prediction_col=self._structure_prediction_col,
            target_col=self._structure_target_col,
        )
        errors_str = format_errors_as_markdown(errors, error_type="structure")
        log_markdown(errors_str, title="Structure Errors")
        return errors_str

    def _find_topic_errors(self, predictions: pl.DataFrame) -> str:
        errors = find_topic_errors(
            predictions=predictions,
            path=self.topic_error_path,
            haiku_col=self._haiku_col,
            topic_col=self._topic_col,
            prediction_col=self._topic_prediction_col,
            target_col=self._topic_target_col,
        )
        errors_str = format_errors_as_markdown(errors, error_type="topic")
        log_markdown(errors_str, title="Topic Errors")
        return errors_str
