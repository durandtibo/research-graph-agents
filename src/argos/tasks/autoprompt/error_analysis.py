r"""Contain code to analyze the errors."""

from __future__ import annotations

__all__ = [
    "find_errors",
    "find_structure_errors",
    "format_errors_as_markdown",
    "format_errors_as_markdown_table",
]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
from iden.io import save_json

from argos.utils.logging import log_markdown

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def find_structure_errors(
    predictions: pl.DataFrame,
    path: Path | None = None,
    target_col: str = "structure_target",
    prediction_col: str = "structure_passed",
) -> list[dict[str, str | bool]]:
    r"""Analyze prediction errors for both structure and topic tasks.

    Finds haiku examples where the judge's predictions do not match
    the ground-truth labels for structure and topic adherence,
    then logs a markdown summary and saves the error details as JSON
    files under ``path``.

    Args:
        predictions: A :class:`~polars.DataFrame` produced by the haiku
            judge, expected to contain the columns ``topic``,
            ``haiku``, ``structure_target``, ``structure_passed``,
            ``topic_target``, and ``topic_passed``.
        path: Directory where the error analysis JSON files are saved.
            Two files are written:
            ``error_analysis_structure.json`` and
            ``error_analysis_topic.json``.
    """
    logger.info("Analyzing structure errors...")
    errors = find_errors(
        predictions=predictions, col_target=target_col, col_prediction=prediction_col
    )
    log_markdown(
        format_errors_as_markdown(errors, error_type="structure"),
        title="Structure Errors",
    )
    if path:
        save_json(errors, path, exist_ok=True)
    return errors


def find_errors(
    predictions: pl.DataFrame, col_target: str, col_prediction: str
) -> list[dict[str, str | bool]]:
    r"""Find haiku examples where the prediction does not match the
    ground-truth label.

    Args:
        predictions: A :class:`~polars.DataFrame` produced by the haiku
            judge, expected to contain the columns ``topic``,
            ``haiku``, ``col_target``, and ``col_prediction``.
        col_target: The column name containing the ground-truth labels.
        col_prediction: The column name containing the predicted labels.

    Returns:
        A list of dicts, one per mispredicted example, each with the
            keys ``topic``, ``haiku``, ``target``, and ``prediction``.
    """
    return (
        predictions.filter(pl.col(col_target) != pl.col(col_prediction))
        .select(["topic", "haiku", col_target, col_prediction])
        .rename({col_target: "target", col_prediction: "prediction"})
        .to_dicts()
    )


def format_errors_as_markdown(errors: list[dict[Any, Any]], error_type: str) -> str:
    r"""Format an error list as a markdown report with a summary header
    and a table.

    Args:
        errors: A list of error dicts as returned by
            :func:`find_errors`, each containing ``topic``, ``haiku``,
            ``target``, and ``prediction``.
        error_type: A human-readable label for the type of error
            (e.g. ``"structure"`` or ``"topic"``), used in the summary
            text and column descriptions.

    Returns:
        A markdown string with a brief summary sentence followed by a
            legend and the formatted error table.
    """
    table = format_errors_as_markdown_table(errors)
    return (
        f"{len(errors)} haikus have incorrect {error_type} predictions. "
        f"The table below details these errors:\n"
        f"- **Topic**: The topic of the haiku (valid only if the topic target is true).\n"
        f"- **Haiku**: The evaluated text, with line breaks (`\\n`) replaced by slashes (` / `)\n"
        f"- **Target**: The true, correct {error_type} label.\n"
        f"- **Prediction**: The model's output {error_type} label.\n"
        f"\n{table}\n"
    )


def format_errors_as_markdown_table(errors: list[dict[Any, Any]]) -> str:
    r"""Format an error list as a markdown table.

    Args:
        errors: A list of error dicts as returned by
            :func:`find_errors`, each containing ``topic``, ``haiku``,
            ``target``, and ``prediction``. Newlines in ``haiku``
            values are replaced with `` / `` for readability.

    Returns:
        A markdown table string with columns ``#``, ``Topic``,
            ``Haiku``, ``Target``, and ``Prediction``.
    """
    lines = ["| # | Topic | Haiku | Target | Prediction |", "|----|----|----|----|----|"]
    for i, example in enumerate(errors, start=1):
        haiku = example["haiku"].replace("\n", " / ")
        lines.append(
            f"| {i} | {example['topic']} | {haiku} | {example['target']} | {example['prediction']} |"
        )
    return "\n".join(lines)
