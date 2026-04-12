from __future__ import annotations

import polars as pl

from argos.autoprompt.haiku import columns
from argos.datasets.haiku import generate_haiku_dataset

##########################################
#     Tests for generate_haiku_dataset   #
##########################################


def test_generate_haiku_dataset_returns_dataframe() -> None:
    assert isinstance(generate_haiku_dataset(), pl.DataFrame)


def test_generate_haiku_dataset_columns() -> None:
    df = generate_haiku_dataset()
    assert df.columns == [
        columns.TOPIC,
        columns.HAIKU,
        columns.STRUCTURE_TARGET,
        columns.TOPIC_TARGET,
        columns.OVERALL_TARGET,
    ]


def test_generate_haiku_dataset_num_rows() -> None:
    df = generate_haiku_dataset()
    assert len(df) == 100


def test_generate_haiku_dataset_schema() -> None:
    df = generate_haiku_dataset()
    assert df.schema == pl.Schema(
        {
            columns.TOPIC: pl.String,
            columns.HAIKU: pl.String,
            columns.STRUCTURE_TARGET: pl.Boolean,
            columns.TOPIC_TARGET: pl.Boolean,
            columns.OVERALL_TARGET: pl.Boolean,
        }
    )


def test_generate_haiku_dataset_num_positive_examples() -> None:
    df = generate_haiku_dataset()
    assert df[columns.OVERALL_TARGET].sum() == 50


def test_generate_haiku_dataset_num_negative_examples() -> None:
    df = generate_haiku_dataset()
    assert (~df[columns.OVERALL_TARGET]).sum() == 50


def test_generate_haiku_dataset_num_correct_structure() -> None:
    df = generate_haiku_dataset()
    assert df[columns.STRUCTURE_TARGET].sum() == 70


def test_generate_haiku_dataset_num_incorrect_structure() -> None:
    df = generate_haiku_dataset()
    assert (~df[columns.STRUCTURE_TARGET]).sum() == 30


def test_generate_haiku_dataset_num_correct_topic() -> None:
    df = generate_haiku_dataset()
    assert df[columns.TOPIC_TARGET].sum() == 70


def test_generate_haiku_dataset_num_incorrect_topic() -> None:
    df = generate_haiku_dataset()
    assert (~df[columns.TOPIC_TARGET]).sum() == 30


def test_generate_haiku_dataset_no_null_values() -> None:
    df = generate_haiku_dataset()
    assert df.null_count().sum_horizontal()[0] == 0
