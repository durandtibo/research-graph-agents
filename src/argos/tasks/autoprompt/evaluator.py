r"""Implement the evaluators."""

from __future__ import annotations

__all__ = ["BaseEvaluator", "HaikuJudgeEvaluator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola.utils.format import repr_indent, repr_mapping
from feu.utils.io import save_json

from argos.tasks.autoprompt.evaluation import evaluate_metrics
from argos.utils.dataclass import dataclass_to_dict

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
        path: An optional path where to save the evaluation metrics.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"path": self._path}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def evaluate(self, predictions: pl.DataFrame) -> dict[str, Any]:
        metrics = dataclass_to_dict(evaluate_metrics(predictions))
        if self._path:
            save_json(metrics, self._path, exist_ok=True)
        return metrics
