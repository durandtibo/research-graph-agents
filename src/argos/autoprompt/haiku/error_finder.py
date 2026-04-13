r"""Contain code to find the errors."""

from __future__ import annotations

__all__ = ["BaseErrorFinder", "ErrorFinder"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

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
    r"""Define the base class to implement an error finder."""

    @abstractmethod
    def find(self, predictions: pl.DataFrame) -> str:
        r"""Find the prediction errors and return a formatted report.

        Args:
            predictions: A DataFrame with the predictions.

        Returns:
            A string report describing the errors found.
        """


class ErrorFinder(BaseErrorFinder):
    r"""Implement a simple error finder.

    Args:
        root_path: The root path where to store the errors.
            If ``None``, no files are written.
        haiku_col: Column name containing the haiku text. Defaults to
            :data:`~argos.autoprompt.haiku.columns.HAIKU`.
        topic_col: Column name containing the topic text. Defaults to
            :data:`~argos.autoprompt.haiku.columns.TOPIC`.
        overall_prediction_col: Column name for the overall predicted
            label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.OVERALL_PREDICTION`.
        overall_target_col: Column name for the overall ground-truth
            label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.OVERALL_TARGET`.
        structure_prediction_col: Column name for the structure
            predicted label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.STRUCTURE_PREDICTION`.
        structure_reasoning_col: Column name for the structure
            reasoning text. Defaults to
            :data:`~argos.autoprompt.haiku.columns.STRUCTURE_REASONING`.
        structure_target_col: Column name for the structure
            ground-truth label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.STRUCTURE_TARGET`.
        topic_prediction_col: Column name for the topic predicted
            label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.TOPIC_PREDICTION`.
        topic_reasoning_col: Column name for the topic reasoning text.
            Defaults to
            :data:`~argos.autoprompt.haiku.columns.TOPIC_REASONING`.
        topic_target_col: Column name for the topic ground-truth
            label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.TOPIC_TARGET`.
    """

    def __init__(
        self,
        root_path: Path | None = None,
        *,
        haiku_col: str = columns.HAIKU,
        topic_col: str = columns.TOPIC,
        overall_prediction_col: str = columns.OVERALL_PREDICTION,
        overall_target_col: str = columns.OVERALL_TARGET,
        structure_prediction_col: str = columns.STRUCTURE_PREDICTION,
        structure_reasoning_col: str = columns.STRUCTURE_REASONING,
        structure_target_col: str = columns.STRUCTURE_TARGET,
        topic_prediction_col: str = columns.TOPIC_PREDICTION,
        topic_reasoning_col: str = columns.TOPIC_REASONING,
        topic_target_col: str = columns.TOPIC_TARGET,
    ) -> None:
        self._root_path = root_path
        self._haiku_col = haiku_col
        self._topic_col = topic_col
        self._overall_prediction_col = overall_prediction_col
        self._overall_target_col = overall_target_col
        self._structure_prediction_col = structure_prediction_col
        self._structure_reasoning_col = structure_reasoning_col
        self._structure_target_col = structure_target_col
        self._topic_prediction_col = topic_prediction_col
        self._topic_reasoning_col = topic_reasoning_col
        self._topic_target_col = topic_target_col

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def structure_error_path(self) -> Path | None:
        r"""The path to the structure error file, or ``None`` if no root
        path was provided."""
        if not self._root_path:
            return None
        return self._root_path.joinpath("error_structure.json")

    @property
    def topic_error_path(self) -> Path | None:
        r"""The path to the topic error file, or ``None`` if no root
        path was provided."""
        if not self._root_path:
            return None
        return self._root_path.joinpath("error_topic.json")

    def find(self, predictions: pl.DataFrame) -> str:
        structure = self._find_structure_errors(predictions)
        topic = self._find_topic_errors(predictions)
        return (
            f"## Examples with Structure Errors\n\n{structure}\n\n"
            f"## Examples with Topic Errors\n\n{topic}"
        )

    def _find_structure_errors(self, predictions: pl.DataFrame) -> str:
        r"""Return a markdown-formatted report of structure prediction
        errors.

        Args:
            predictions: A :class:`~polars.DataFrame` with the
                predictions.

        Returns:
            A markdown string summarising the structure errors.
        """
        errors = find_structure_errors(
            predictions=predictions,
            path=self.structure_error_path,
            haiku_col=self._haiku_col,
            topic_col=self._topic_col,
            prediction_col=self._structure_prediction_col,
            reasoning_col=self._structure_reasoning_col,
            target_col=self._structure_target_col,
        )
        errors_str = format_errors_as_markdown(errors, error_type="structure")
        log_markdown(errors_str, title="Structure Errors")
        return errors_str

    def _find_topic_errors(self, predictions: pl.DataFrame) -> str:
        r"""Return a markdown-formatted report of topic prediction
        errors.

        Args:
            predictions: A :class:`~polars.DataFrame` with the
                predictions.

        Returns:
            A markdown string summarising the topic errors.
        """
        errors = find_topic_errors(
            predictions=predictions,
            path=self.topic_error_path,
            haiku_col=self._haiku_col,
            topic_col=self._topic_col,
            prediction_col=self._topic_prediction_col,
            reasoning_col=self._topic_reasoning_col,
            target_col=self._topic_target_col,
        )
        errors_str = format_errors_as_markdown(errors, error_type="topic")
        log_markdown(errors_str, title="Topic Errors")
        return errors_str

    def _get_kwargs(self) -> dict[str, Any]:
        return {
            "path": self._root_path,
            "haiku_col": self._haiku_col,
            "topic_col": self._topic_col,
            "overall_prediction_col": self._overall_prediction_col,
            "overall_target_col": self._overall_target_col,
            "structure_prediction_col": self._structure_prediction_col,
            "structure_reasoning_col": self._structure_reasoning_col,
            "structure_target_col": self._structure_target_col,
            "topic_prediction_col": self._topic_prediction_col,
            "topic_reasoning_col": self._topic_reasoning_col,
            "topic_target_col": self._topic_target_col,
        }
