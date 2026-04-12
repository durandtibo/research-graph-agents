r"""Implement the evaluators."""

from __future__ import annotations

__all__ = ["BaseEvaluator", "HaikuJudgeEvaluator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola.utils.format import repr_indent, repr_mapping
from feu.utils.io import save_json

from argos.autoprompt.haiku import columns
from argos.autoprompt.haiku.evaluation import evaluate_judge_classification_metrics
from argos.utils.mapping import recursive_to_dict

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl


class BaseEvaluator(ABC):
    r"""Abstract base class for evaluators."""

    @abstractmethod
    def evaluate(self, predictions: pl.DataFrame) -> dict[str, Any]:
        r"""Evaluate the performances.

        Args:
            predictions: The predictions and targets.

        Returns:
            The evaluation results.
        """


class HaikuJudgeEvaluator(BaseEvaluator):
    r"""Evaluate the performances of the Haiku Judge.

    Args:
        path: An optional path where to save the evaluation metrics as
            a JSON file. If ``None``, the metrics are not persisted.
        overall_prediction_col: Column name for the overall predicted
            label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.OVERALL_PREDICTION`.
        overall_target_col: Column name for the overall ground-truth
            label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.OVERALL_TARGET`.
        structure_prediction_col: Column name for the structure
            predicted label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.STRUCTURE_PREDICTION`.
        structure_target_col: Column name for the structure
            ground-truth label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.STRUCTURE_TARGET`.
        topic_prediction_col: Column name for the topic predicted
            label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.TOPIC_PREDICTION`.
        topic_target_col: Column name for the topic ground-truth
            label. Defaults to
            :data:`~argos.autoprompt.haiku.columns.TOPIC_TARGET`.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        overall_prediction_col: str = columns.OVERALL_PREDICTION,
        overall_target_col: str = columns.OVERALL_TARGET,
        structure_prediction_col: str = columns.STRUCTURE_PREDICTION,
        structure_target_col: str = columns.STRUCTURE_TARGET,
        topic_prediction_col: str = columns.TOPIC_PREDICTION,
        topic_target_col: str = columns.TOPIC_TARGET,
    ) -> None:
        self._path = path
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

    def evaluate(self, predictions: pl.DataFrame) -> dict[str, Any]:
        metrics = evaluate_judge_classification_metrics(
            predictions,
            overall_prediction_col=self._overall_prediction_col,
            overall_target_col=self._overall_target_col,
            structure_prediction_col=self._structure_prediction_col,
            structure_target_col=self._structure_target_col,
            topic_prediction_col=self._topic_prediction_col,
            topic_target_col=self._topic_target_col,
        )
        metrics = recursive_to_dict(metrics)
        if self._path:
            save_json(metrics, self._path, exist_ok=True)
        return metrics
