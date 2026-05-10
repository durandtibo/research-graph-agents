r"""Contain a utility function to prepare the haiku judge dataset.

This module loads the labeled haiku dataset and logs summary statistics
about the boolean target columns.
"""

from __future__ import annotations

__all__ = ["prepare_dataset"]

import logging

import polars as pl
from coola.utils.timing import timeblock

from argos.autoprompt.haiku import columns
from argos.datasets import generate_haiku_dataset
from argos.utils.dataframe import summarize_boolean_columns

logger: logging.Logger = logging.getLogger(__name__)


def prepare_dataset() -> pl.DataFrame:
    r"""Prepare a dataset of haiku examples.

    Returns:
        A :class:`~polars.DataFrame` with columns ``topic``,
            ``haiku``, ``structure_target``, ``topic_target``, and
            ``overall_target`` — matching the schema produced by
            :func:`~argos.datasets.generate_haiku_dataset`.
    """
    logger.info("Preparing dataset...")
    with timeblock(message="Dataset generation time: {time}"):
        dataset = generate_haiku_dataset()

    # uncomment this line to sample a smaller version of the dataset.
    # dataset = dataset.sample(n=5, seed=42)
    with pl.Config(tbl_cols=-1, tbl_rows=10):
        logger.info(f"\n{dataset}")

    stats = summarize_boolean_columns(
        dataset.select([columns.OVERALL_TARGET, columns.STRUCTURE_TARGET, columns.TOPIC_TARGET])
    )
    logger.info(f"statistics about the dataset\n{stats}")
    return dataset
