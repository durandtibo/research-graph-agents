from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from argos.meta_agent.result import Result
from argos.meta_agent.result.utils import results_to_dataframe

##########################################
#     Tests for results_to_dataframe     #
##########################################


def test_results_to_dataframe_empty() -> None:
    frame = results_to_dataframe([])
    assert frame.is_empty()


def test_results_to_dataframe_single_result() -> None:
    frame = results_to_dataframe([Result({"loss": 0.5, "accuracy": 0.9})])
    assert_frame_equal(frame, pl.DataFrame({"loss": [0.5], "accuracy": [0.9]}))


def test_results_to_dataframe_multiple_results() -> None:
    frame = results_to_dataframe(
        [
            Result({"loss": 0.5, "accuracy": 0.9}),
            Result({"loss": 0.3, "accuracy": 0.95}),
        ]
    )
    assert_frame_equal(frame, pl.DataFrame({"loss": [0.5, 0.3], "accuracy": [0.9, 0.95]}))


def test_results_to_dataframe_single_metric() -> None:
    frame = results_to_dataframe(
        [
            Result({"loss": 0.5}),
            Result({"loss": 0.3}),
        ]
    )
    assert_frame_equal(frame, pl.DataFrame({"loss": [0.5, 0.3]}))


def test_results_to_dataframe_row_count() -> None:
    frame = results_to_dataframe([Result({"loss": float(i)}) for i in range(5)])
    assert frame.shape[0] == 5


def test_results_to_dataframe_integer_values() -> None:
    frame = results_to_dataframe([Result({"epoch": 1, "step": 100})])
    expected = pl.DataFrame({"epoch": [1], "step": [100]})
    assert_frame_equal(frame, expected)


def test_results_to_dataframe_string_values() -> None:
    frame = results_to_dataframe([Result({"model": "resnet50"})])
    expected = pl.DataFrame({"model": ["resnet50"]})
    assert_frame_equal(frame, expected)
