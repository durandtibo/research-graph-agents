r"""Contain the code to evaluate the performances."""

from __future__ import annotations

__all__ = ["evaluate_metrics"]

import logging
from typing import TYPE_CHECKING

from argos.metrics import (
    BinaryClassificationResults,
    compute_binary_classification_metrics,
)

if TYPE_CHECKING:
    import polars as pl

logger: logging.Logger = logging.getLogger(__name__)


def evaluate_metrics(predictions: pl.DataFrame) -> dict[str, BinaryClassificationResults]:
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
        predictions, target_col="target", prediction_col="passed"
    )
    logger.info(f"overall\n{overall.to_str()}")

    structure = compute_binary_classification_metrics(
        predictions, target_col="structure_target", prediction_col="structure_passed"
    )
    logger.info(f"structure\n{structure.to_str()}")

    topic = compute_binary_classification_metrics(
        predictions, target_col="topic_target", prediction_col="topic_passed"
    )
    logger.info(f"topic\n{topic.to_str()}")
    return {"overall": overall, "structure": structure, "topic": topic}
