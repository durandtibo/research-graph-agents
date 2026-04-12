r"""Contain prediction error analysis utilities.

This module provides functions to identify haiku examples where the
judge's predictions disagree with the ground-truth labels, and to format
those mismatches as human-readable markdown reports.
"""

import logging
from pathlib import Path

import polars as pl
from iden.io import save_json

from argos.autoprompt.haiku.error_analysis import (
    find_errors,
    format_errors_as_markdown,
)
from argos.utils.logging import log_markdown

logger: logging.Logger = logging.getLogger(__name__)


def analyze_errors(results: pl.DataFrame, path: Path) -> None:
    r"""Analyze prediction errors for both structure and topic
    autoprompt.

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
    logger.info("Analyzing structure errors...")
    structure_errors = find_errors(
        predictions=results, target_col="structure_target", prediction_col="structure_passed"
    )
    log_markdown(
        format_errors_as_markdown(structure_errors, error_type="structure"),
        title="Structure Errors",
    )
    save_json(structure_errors, path.joinpath("error_analysis_structure.json"), exist_ok=True)

    logger.info("Analyzing topic errors...")
    topic_errors = find_errors(
        predictions=results, target_col="topic_target", prediction_col="topic_passed"
    )
    log_markdown(format_errors_as_markdown(topic_errors, error_type="topic"), title="Topic Errors")
    save_json(topic_errors, path.joinpath("error_analysis_topic.json"), exist_ok=True)
