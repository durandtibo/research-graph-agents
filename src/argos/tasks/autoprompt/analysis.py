r"""Contain code to run the autoprompt on the haiku dataset."""

from __future__ import annotations

__all__ = [
    "analyze_errors",
    "find_errors",
    "format_errors_as_markdown",
    "format_errors_as_markdown_table",
]

import logging
from typing import TYPE_CHECKING

import polars as pl
from iden.io import save_json

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
        .select(["topic", "haiku", col_target, col_prediction])
        .rename({col_target: "target", col_prediction: "prediction"})
        .to_dicts()
    )


def format_errors_as_markdown(errors: list[dict], error_type: str) -> str:
    r"""Format the list of errors to a markdown table.

    Args:
        errors: The list of errors.
        error_type: The type of errors.

    Returns:
        The formatted list of errors to a markdown table.
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


def format_errors_as_markdown_table(errors: list[dict]) -> str:
    r"""Format the list of errors to a markdown table.

    Args:
        errors: The list of errors.

    Returns:
        The formatted list of errors to a markdown table.
    """
    lines = ["| # | Topic | Haiku | Target | Prediction |", "|----|----|----|----|----|"]
    for i, example in enumerate(errors, start=1):
        haiku = example["haiku"].replace("\n", " / ")
        lines.append(
            f"| {i} | {example['topic']} | {haiku} | {example['target']} | {example['prediction']} |"
        )
    return "\n".join(lines)
