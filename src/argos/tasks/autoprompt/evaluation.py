r"""Contain the code to evaluate the performances."""

from __future__ import annotations

__all__ = ["evaluate_metrics"]

import logging
from typing import TYPE_CHECKING

from argos.metrics import (
    BinaryClassificationResults,
    compute_binary_classification_metrics,
)
from argos.utils.dataframe import summarize_boolean_columns

if TYPE_CHECKING:
    import polars as pl

logger: logging.Logger = logging.getLogger(__name__)


def evaluate_metrics(results: pl.DataFrame) -> dict[str, BinaryClassificationResults]:
    r"""Evaluate the metrics of the haiku judge.

    Args:
        results: The results of the haiku judge.

    Returns:
        The evaluated metrics.
    """
    logger.info("Evaluating metrics...")
    logger.info(
        f"\n{summarize_boolean_columns(results.select(['target', 'structure_target', 'topic_target']))}"
    )

    overall = compute_binary_classification_metrics(
        results, target_col="target", predict_col="passed"
    )
    logger.info(f"overall\n{overall.to_str()}")

    structure = compute_binary_classification_metrics(
        results, target_col="structure_target", predict_col="structure_passed"
    )
    logger.info(f"structure\n{structure.to_str()}")

    topic = compute_binary_classification_metrics(
        results, target_col="topic_target", predict_col="topic_passed"
    )
    logger.info(f"topic\n{topic.to_str()}")
    return {"overall": overall, "structure": structure, "topic": topic}
