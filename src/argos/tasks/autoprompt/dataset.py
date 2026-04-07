r"""Contain code to prepare the datasets."""

from __future__ import annotations

__all__ = ["prepare_dataset"]

import polars as pl
from coola.utils.timing import timeblock

from argos.datasets import generate_haiku_dataset
from argos.tasks.autoprompt.judge import logger
from argos.utils.dataframe import summarize_boolean_columns


def prepare_dataset() -> pl.DataFrame:
    r"""Prepare a dataset of haiku examples.

    Returns:
        A DataFrame containing haiku examples.
    """
    logger.info("Preparing dataset...")
    with timeblock(message="Dataset generation time: {time}"):
        dataset = generate_haiku_dataset()

    # uncomment this line to sample a smaller version of the dataset.
    # dataset = dataset.sample(n=5, seed=42)
    with pl.Config(tbl_cols=-1, tbl_rows=10):
        logger.info(f"\n{dataset}")

    stats = summarize_boolean_columns(
        dataset.select(["target", "structure_target", "topic_target"])
    )
    logger.info(f"statistics about the dataset\n{stats}")
    return dataset
