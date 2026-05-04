from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from argos.meta_agent.results import Result, ResultDict
from argos.meta_agent.results.utils import results_to_dataframe

##########################################
#     Tests for results_to_dataframe     #
##########################################


def test_results_to_dataframe_empty() -> None:
    frame = results_to_dataframe([])
    assert_frame_equal(frame, pl.DataFrame({}))


def test_results_to_dataframe_single_result() -> None:
    frame = results_to_dataframe([Result({"loss": 0.5, "accuracy": 0.9})])
    assert_frame_equal(
        frame,
        pl.DataFrame(
            {"loss": [0.5], "accuracy": [0.9]}, schema={"loss": pl.Float64, "accuracy": pl.Float64}
        ),
    )


def test_results_to_dataframe_multiple_results() -> None:
    frame = results_to_dataframe(
        [
            Result({"loss": 0.5, "accuracy": 0.9}),
            Result({"loss": 0.3, "accuracy": 0.95}),
        ]
    )
    assert_frame_equal(
        frame,
        pl.DataFrame(
            {"loss": [0.5, 0.3], "accuracy": [0.9, 0.95]},
            schema={"loss": pl.Float64, "accuracy": pl.Float64},
        ),
    )


def test_results_to_dataframe_single_metric() -> None:
    frame = results_to_dataframe([Result({"loss": 0.5}), Result({"loss": 0.3})])
    assert_frame_equal(frame, pl.DataFrame({"loss": [0.5, 0.3]}, schema={"loss": pl.Float64}))


def test_results_to_dataframe_multiple_types() -> None:
    frame = results_to_dataframe(
        [
            Result({"model": "resnet50", "loss": 0.5, "rank": 1}),
            Result({"model": "resnet30", "loss": 0.3, "rank": 2}),
        ]
    )
    assert_frame_equal(
        frame,
        pl.DataFrame(
            {"model": ["resnet50", "resnet30"], "loss": [0.5, 0.3], "rank": [1, 2]},
            schema={"model": pl.String, "loss": pl.Float64, "rank": pl.Int64},
        ),
    )


def test_results_to_dataframe_with_result_dict() -> None:
    frame = results_to_dataframe(
        [
            ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})}),
            ResultDict({"train": Result({"loss": 0.4}), "val": Result({"loss": 0.2})}),
        ]
    )
    assert_frame_equal(
        frame,
        pl.DataFrame(
            {"train.loss": [0.5, 0.4], "val.loss": [0.3, 0.2]},
            schema={"train.loss": pl.Float64, "val.loss": pl.Float64},
        ),
    )
