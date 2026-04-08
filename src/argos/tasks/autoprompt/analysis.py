r"""Contain code to run the autoprompt on the haiku dataset."""

from __future__ import annotations

__all__ = [
    "analyze_errors",
    "find_errors",
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


def analyze_errors(results: pl.DataFrame, path: Path) -> None:
    r"""Analyze prediction errors for both structure and topic tasks.

    Finds haiku examples where the judge's predictions do not match
    the ground-truth labels for structure and topic adherence,
    then logs a markdown summary and saves the error details as JSON
    files under ``path``.

    Args:
        results: A :class:`~polars.DataFrame` produced by the haiku
            judge, expected to contain the columns ``topic``,
            ``haiku``, ``structure_target``, ``structure_passed``,
            ``topic_target``, and ``topic_passed``.
        path: Directory where the error analysis JSON files are saved.
            Two files are written:
            ``error_analysis_structure.json`` and
            ``error_analysis_topic.json``.
    """
    logger.info("Haikus with incorrect structure predictions")
    structure_errors = find_errors(
        results=results, col_target="structure_target", col_prediction="structure_passed"
    )
    log_markdown(format_errors_as_markdown(structure_errors, error_type="structure"))
    save_json(structure_errors, path.joinpath("error_analysis_structure.json"), exist_ok=True)

    logger.info("Haikus with incorrect topic predictions")
    topic_errors = find_errors(
        results=results, col_target="topic_target", col_prediction="topic_passed"
    )
    log_markdown(format_errors_as_markdown(topic_errors, error_type="topic"))
    save_json(topic_errors, path.joinpath("error_analysis_topic.json"), exist_ok=True)


def find_errors(
    results: pl.DataFrame, col_target: str, col_prediction: str
) -> list[dict[str, str | bool]]:
    r"""Find haiku examples where the prediction does not match the
    ground-truth label.

    Args:
        results: A :class:`~polars.DataFrame` produced by the haiku
            judge, expected to contain the columns ``topic``,
            ``haiku``, ``col_target``, and ``col_prediction``.
        col_target: The column name containing the ground-truth labels.
        col_prediction: The column name containing the predicted labels.

    Returns:
        A list of dicts, one per mispredicted example, each with the
            keys ``topic``, ``haiku``, ``target``, and ``prediction``.
    """
    return (
        results.filter(pl.col(col_target) != pl.col(col_prediction))
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
