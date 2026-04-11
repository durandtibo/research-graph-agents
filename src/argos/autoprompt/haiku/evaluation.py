r"""Contain the code to evaluate the performances."""

from __future__ import annotations

__all__ = ["evaluate_judge_classification_metrics"]

import logging
from typing import TYPE_CHECKING

from argos.autoprompt.haiku import columns
from argos.metrics import (
    BinaryClassificationResults,
    compute_binary_classification_metrics,
)

if TYPE_CHECKING:
    import polars as pl

logger: logging.Logger = logging.getLogger(__name__)


def evaluate_judge_classification_metrics(
    predictions: pl.DataFrame,
    *,
    overall_prediction_col: str = columns.OVERALL_PREDICTION,
    overall_target_col: str = columns.OVERALL_TARGET,
    structure_prediction_col: str = columns.STRUCTURE_PREDICTION,
    structure_target_col: str = columns.STRUCTURE_TARGET,
    topic_prediction_col: str = columns.TOPIC_PREDICTION,
    topic_target_col: str = columns.TOPIC_TARGET,
) -> dict[str, BinaryClassificationResults]:
    r"""Evaluate the metrics of the haiku judge.

    Computes binary classification metrics for three prediction autoprompt:
    overall pass/fail, structure adherence, and topic relevance.

    Args:
        predictions: A :class:`~polars.DataFrame` produced by the haiku
            judge, expected to contain the columns ``target``,
            ``passed``, ``structure_target``, ``structure_passed``,
            ``topic_target``, and ``topic_passed``.

    Returns:
        A dict with three keys mapping to
            :class:`~argos.metrics.BinaryClassificationResults`:

            - ``"overall"``: metrics comparing ``target`` vs ``passed``.
            - ``"structure"``: metrics comparing ``structure_target``
              vs ``structure_passed``.
            - ``"topic"``: metrics comparing ``topic_target`` vs
              ``topic_passed``.
    """
    logger.info("Evaluating metrics...")
    overall = compute_binary_classification_metrics(
        predictions, target_col=overall_target_col, prediction_col=overall_prediction_col
    )
    logger.info(f"overall\n{overall.to_str()}")

    structure = compute_binary_classification_metrics(
        predictions, target_col=structure_target_col, prediction_col=structure_prediction_col
    )
    logger.info(f"structure\n{structure.to_str()}")

    topic = compute_binary_classification_metrics(
        predictions, target_col=topic_target_col, prediction_col=topic_prediction_col
    )
    logger.info(f"topic\n{topic.to_str()}")
    return {"overall": overall, "structure": structure, "topic": topic}
