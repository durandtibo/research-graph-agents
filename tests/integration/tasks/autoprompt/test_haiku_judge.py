from __future__ import annotations

import polars as pl

from argos.tasks.autoprompt.haiku_judge import prepare_dataset

#####################################
#     Tests for prepare_dataset     #
#####################################


def test_prepare_dataset_returns_dataframe() -> None:
    assert isinstance(prepare_dataset(), pl.DataFrame)


def test_prepare_dataset_columns() -> None:
    df = prepare_dataset()
    assert df.columns == ["topic", "haiku", "structure_target", "topic_target", "target"]


def test_prepare_dataset_num_rows() -> None:
    df = prepare_dataset()
    assert len(df) == 100


def test_prepare_dataset_schema() -> None:
    df = prepare_dataset()
    assert df.schema == pl.Schema(
        {
            "topic": pl.String,
            "haiku": pl.String,
            "structure_target": pl.Boolean,
            "topic_target": pl.Boolean,
            "target": pl.Boolean,
        }
    )
