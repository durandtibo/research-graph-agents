from __future__ import annotations

import polars as pl

from argos.datasets.haiku import generate_haiku_dataset

##########################################
#     Tests for generate_haiku_dataset   #
##########################################


def test_generate_haiku_dataset_returns_dataframe() -> None:
    assert isinstance(generate_haiku_dataset(), pl.DataFrame)


def test_generate_haiku_dataset_columns() -> None:
    df = generate_haiku_dataset()
    assert df.columns == ["topic", "haiku", "structure_target", "topic_target", "target"]


def test_generate_haiku_dataset_num_rows() -> None:
    df = generate_haiku_dataset()
    assert len(df) == 100


def test_generate_haiku_dataset_schema() -> None:
    df = generate_haiku_dataset()
    assert df.schema == pl.Schema(
        {
            "topic": pl.String,
            "haiku": pl.String,
            "structure_target": pl.Boolean,
            "topic_target": pl.Boolean,
            "target": pl.Boolean,
        }
    )


def test_generate_haiku_dataset_num_positive_examples() -> None:
    df = generate_haiku_dataset()
    assert df["target"].sum() == 50


def test_generate_haiku_dataset_num_negative_examples() -> None:
    df = generate_haiku_dataset()
    assert (~df["target"]).sum() == 50


def test_generate_haiku_dataset_num_correct_structure() -> None:
    df = generate_haiku_dataset()
    assert df["structure_target"].sum() == 70


def test_generate_haiku_dataset_num_incorrect_structure() -> None:
    df = generate_haiku_dataset()
    assert (~df["structure_target"]).sum() == 30


def test_generate_haiku_dataset_num_correct_topic() -> None:
    df = generate_haiku_dataset()
    assert df["topic_target"].sum() == 70


def test_generate_haiku_dataset_num_incorrect_topic() -> None:
    df = generate_haiku_dataset()
    assert (~df["topic_target"]).sum() == 30


def test_generate_haiku_dataset_no_null_values() -> None:
    df = generate_haiku_dataset()
    assert df.null_count().sum_horizontal()[0] == 0
