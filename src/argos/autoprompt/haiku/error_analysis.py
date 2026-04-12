r"""Contain code to analyze the errors."""

from __future__ import annotations

__all__ = [
    "find_errors",
    "find_structure_errors",
    "find_topic_errors",
    "format_errors_as_markdown",
    "format_errors_as_markdown_table",
]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
from iden.io import save_json

from argos.autoprompt.haiku import columns
from argos.utils.logging import log_markdown

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def find_structure_errors(
    predictions: pl.DataFrame,
    path: Path | None = None,
    *,
    haiku_col: str = columns.HAIKU,
    topic_col: str = columns.TOPIC,
    prediction_col: str = columns.STRUCTURE_PREDICTION,
    target_col: str = columns.STRUCTURE_TARGET,
) -> list[dict[str, str | bool]]:
    r"""Find haiku examples where the structure prediction does not match
    the ground-truth label, log a markdown summary, and optionally save
    the results.

    Args:
        predictions: A :class:`~polars.DataFrame` produced by the haiku
            judge, expected to contain the columns identified by
            ``topic_col``, ``haiku_col``, ``target_col``, and
            ``prediction_col``.
        path: Optional path where the error list is saved as a JSON
            file. If ``None``, no file is written.
        haiku_col: The column name containing the haiku text.
            Defaults to
            :data:`~argos.autoprompt.haiku.columns.HAIKU`.
        topic_col: The column name containing the topic text.
            Defaults to
            :data:`~argos.autoprompt.haiku.columns.TOPIC`.
        prediction_col: The column name containing the predicted
            structure labels. Defaults to
            :data:`~argos.autoprompt.haiku.columns.STRUCTURE_PREDICTION`.
        target_col: The column name containing the ground-truth
            structure labels. Defaults to
            :data:`~argos.autoprompt.haiku.columns.STRUCTURE_TARGET`.

    Returns:
        A list of dicts, one per mispredicted example, each with the
            keys ``topic``, ``haiku``, ``target``, and ``prediction``.
    """
    logger.info("Analyzing structure errors...")
    errors = find_errors(
        predictions=predictions,
        target_col=target_col,
        prediction_col=prediction_col,
        haiku_col=haiku_col,
        topic_col=topic_col,
    )
    log_markdown(
        format_errors_as_markdown(errors, error_type="structure"),
        title="Structure Errors",
    )
    if path:
        save_json(errors, path, exist_ok=True)
    return errors


def find_topic_errors(
    predictions: pl.DataFrame,
    path: Path | None = None,
    *,
    haiku_col: str = columns.HAIKU,
    topic_col: str = columns.TOPIC,
    prediction_col: str = columns.TOPIC_PREDICTION,
    target_col: str = columns.TOPIC_TARGET,
) -> list[dict[str, str | bool]]:
    r"""Find haiku examples where the topic prediction does not match the
    ground-truth label, log a markdown summary, and optionally save the
    results.

    Args:
        predictions: A :class:`~polars.DataFrame` produced by the haiku
            judge, expected to contain the columns identified by
            ``topic_col``, ``haiku_col``, ``target_col``, and
            ``prediction_col``.
        path: Optional path where the error list is saved as a JSON
            file. If ``None``, no file is written.
        haiku_col: The column name containing the haiku text.
            Defaults to
            :data:`~argos.autoprompt.haiku.columns.HAIKU`.
        topic_col: The column name containing the topic text.
            Defaults to
            :data:`~argos.autoprompt.haiku.columns.TOPIC`.
        prediction_col: The column name containing the predicted
            topic labels. Defaults to
            :data:`~argos.autoprompt.haiku.columns.TOPIC_PREDICTION`.
        target_col: The column name containing the ground-truth
            topic labels. Defaults to
            :data:`~argos.autoprompt.haiku.columns.TOPIC_TARGET`.

    Returns:
        A list of dicts, one per mispredicted example, each with the
            keys ``topic``, ``haiku``, ``target``, and ``prediction``.
    """
    logger.info("Analyzing topic errors...")
    errors = find_errors(
        predictions=predictions,
        target_col=target_col,
        prediction_col=prediction_col,
        haiku_col=haiku_col,
        topic_col=topic_col,
    )
    log_markdown(
        format_errors_as_markdown(errors, error_type="topic"),
        title="Topic Errors",
    )
    if path:
        save_json(errors, path, exist_ok=True)
    return errors


def find_errors(
    predictions: pl.DataFrame,
    *,
    prediction_col: str,
    target_col: str,
    haiku_col: str = columns.HAIKU,
    topic_col: str = columns.TOPIC,
) -> list[dict[str, str | bool]]:
    r"""Find haiku examples where the prediction does not match the
    ground-truth label.

    Args:
        predictions: A :class:`~polars.DataFrame` produced by the haiku
            judge, expected to contain the columns identified by
            ``topic_col``, ``haiku_col``, ``target_col``, and
            ``prediction_col``.
        prediction_col: The column name containing the predicted labels.
        target_col: The column name containing the ground-truth labels.
        haiku_col: The column name containing the haiku text.
            Defaults to
            :data:`~argos.autoprompt.haiku.columns.HAIKU`.
        topic_col: The column name containing the topic text.
            Defaults to
            :data:`~argos.autoprompt.haiku.columns.TOPIC`.

    Returns:
        A list of dicts, one per mispredicted example, each with the
            keys ``topic``, ``haiku``, ``target``, and ``prediction``.
    """
    return (
        predictions.filter(pl.col(target_col) != pl.col(prediction_col))
        .select([topic_col, haiku_col, target_col, prediction_col])
        .rename({target_col: "target", prediction_col: "prediction"})
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
        haiku = example[columns.HAIKU].replace("\n", " / ")
        lines.append(
            f"| {i} | {example[columns.TOPIC]} | {haiku} | {example[columns.TARGET]} "
            f"| {example[columns.PREDICTION]} |"
        )
    return "\n".join(lines)
