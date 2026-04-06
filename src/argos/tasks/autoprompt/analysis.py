r"""Contain code to run the autoprompt on the haiku dataset."""

from __future__ import annotations

__all__ = [
    "analyze_errors",
    "find_errors",
    "format_errors_as_markdown_table",
    "format_incorrect_structure_haiku",
    "format_incorrect_topic_haiku",
]

import logging
from typing import TYPE_CHECKING

import polars as pl
from iden.io import save_text

from argos.utils.logging import log_markdown

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def analyze_errors(results: pl.DataFrame, path: Path) -> None:
    r"""Analyze the haikus with errors and save the results in markdown
    files.

    Args:
        results: The results of the haiku judge.
        path: The path where to store the analyses.
    """
    logger.info("Haikus with incorrect structure predictions")
    structure_errors = find_errors(
        results=results, col_target="structure_target", col_prediction="structure_passed"
    )
    log_markdown(format_errors_as_markdown_table(structure_errors))
    save_text(structure_errors, path.joinpath("error_analysis_structure.json"))

    logger.info("Haikus with incorrect topic predictions")
    topic = format_incorrect_topic_haiku(results)
    log_markdown(topic)
    save_text(topic, path.joinpath("error_analysis_topic.md"))


def find_errors(
    results: pl.DataFrame, col_target: str, col_prediction: str
) -> list[dict[str, str | bool]]:
    r"""Find the haiku with incorrect predictions.

    Args:
        results: The results of the haiku judge.
        col_target: The column name of the targets.
        col_prediction: The column name of the predictions.

    Returns:
        The list of haikus with incorrect predictions.
    """
    return (
        results.filter(pl.col(col_target) != pl.col(col_prediction))
        .select(["haiku", col_target, col_prediction])
        .rename({col_target: "target", col_prediction: "prediction"})
        .to_dicts()
    )


def format_incorrect_structure_haiku(results: pl.DataFrame) -> str:
    r"""Format the list of haikus with incorrect structure predictions.

    Args:
        results: The results of the haiku judge.

    Returns:
        The formatted list of haikus with incorrect structure predictions.
    """
    haikus = find_errors(
        results=results, col_target="structure_target", col_prediction="structure_passed"
    )
    haikus_str = "\n".join(map(str, haikus))
    return (
        f"{len(haikus)} haikus have incorrect structure predictions. "
        f"Here is the list of haikus with the true label (target) "
        f"and the predicted label (target):\n"
        f"```jsonl\n\n{haikus_str}\n\n```\n"
    )


def format_incorrect_topic_haiku(results: pl.DataFrame) -> str:
    r"""Format the list of haikus with incorrect topic predictions.

    Args:
        results: The results of the haiku judge.

    Returns:
        The formatted list of haikus with incorrect topic predictions.
    """
    haikus = find_errors(results=results, col_target="topic_target", col_prediction="topic_passed")
    haikus_str = "\n".join(map(str, haikus))
    return (
        f"{len(haikus)} haikus have incorrect topic predictions. "
        f"Here is the list of haikus with the true label (target) "
        f"and the predicted label (target):\n"
        f"```json\n[\n{haikus_str}\n]\n```\n"
    )


def format_errors_as_markdown_table(errors: list[dict]) -> str:
    r"""Format the list of errors to a markdown table.

    Args:
        errors: The list of errors.

    Returns:
        The formatted list of errors to a markdown table.
    """
    lines = ["| # | Haiku | Target | Prediction |", "|----|----|----|----|"]
    for i, example in enumerate(errors, start=1):
        haiku = example["haiku"].replace("\n", " / ")
        lines.append(f"| {i} | {haiku} | {example['target']} | {example['prediction']} |")
    return "\n".join(lines)
