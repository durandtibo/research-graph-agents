r"""Contain code to run the autoprompt on the haiku dataset."""

from __future__ import annotations

__all__ = [
    "analyze_errors",
    "find_incorrect_predictions",
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
    structure = format_incorrect_structure_haiku(results)
    log_markdown(structure)
    save_text(structure, path.joinpath("error_analysis_structure.md"))

    logger.info("Haikus with incorrect topic predictions")
    topic = format_incorrect_topic_haiku(results)
    log_markdown(topic)
    save_text(topic, path.joinpath("error_analysis_topic.md"))


def find_incorrect_predictions(
    results: pl.DataFrame, col_target: str, col_prediction: str
) -> list[str]:
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
        .select(
            pl.format(
                "(haiku): {} (target): {} (prediction): {}",
                pl.col("haiku"),
                pl.col(col_target),
                pl.col(col_prediction),
            )
        )
        .to_series()
        .to_list()
    )


def format_incorrect_structure_haiku(results: pl.DataFrame) -> str:
    r"""Format the list of haikus with incorrect structure predictions.

    Args:
        results: The results of the haiku judge.

    Returns:
        The formatted list of haikus with incorrect structure predictions.
    """
    haikus = find_incorrect_predictions(
        results=results, col_target="structure_target", col_prediction="structure_passed"
    )
    haikus_str = "\n- ".join(["", *haikus])
    return (
        f"{len(haikus)} haikus have incorrect structure predictions. "
        f"Here is the list of haikus with the true label (target) "
        f"and the predicted label (target):{haikus_str}\n"
    )


def format_incorrect_topic_haiku(results: pl.DataFrame) -> str:
    r"""Format the list of haikus with incorrect topic predictions.

    Args:
        results: The results of the haiku judge.

    Returns:
        The formatted list of haikus with incorrect topic predictions.
    """
    haikus = find_incorrect_predictions(
        results=results, col_target="topic_target", col_prediction="topic_passed"
    )
    haikus_str = "\n- ".join(["", *haikus])
    return (
        f"{len(haikus)} haikus have incorrect topic predictions. "
        f"Here is the list of haikus with the true label (target) "
        f"and the predicted label (target):{haikus_str}\n"
    )
