from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import polars as pl
import pytest
from coola.equality.handler.polars import assert_frame_equal

from argos.autoprompt.haiku.inference import InferencePipeline
from argos.autoprompt.haiku.predictor import BasePredictor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def mock_dataset() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
        ]
    )


@pytest.fixture
def mock_results() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "score": 10,
                "passed": True,
                "target": True,
                "structure_passed": True,
                "structure_target": True,
                "topic_passed": True,
                "topic_target": True,
                "reasoning": "reason1",
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                "score": 9,
                "passed": True,
                "target": True,
                "structure_passed": True,
                "structure_target": True,
                "topic_passed": True,
                "topic_target": True,
                "reasoning": "reason2",
            },
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "score": 8,
                "passed": True,
                "target": True,
                "structure_passed": True,
                "structure_target": True,
                "topic_passed": True,
                "topic_target": True,
                "reasoning": "reason3",
            },
        ]
    )


#######################################
#     Tests for InferencePipeline     #
#######################################


def test_inference_pipeline_repr(mock_dataset: pl.DataFrame, mock_results: pl.DataFrame) -> None:
    inferpipe = InferencePipeline(
        dataset=mock_dataset,
        predictor=Mock(spec=BasePredictor, predict=Mock(return_value=mock_results)),
    )
    assert repr(inferpipe).startswith("InferencePipeline(")


def test_inference_pipeline_str(mock_dataset: pl.DataFrame, mock_results: pl.DataFrame) -> None:
    inferpipe = InferencePipeline(
        dataset=mock_dataset,
        predictor=Mock(spec=BasePredictor, predict=Mock(return_value=mock_results)),
    )
    assert str(inferpipe).startswith("InferencePipeline(")


def test_inference_pipeline_process(mock_dataset: pl.DataFrame, mock_results: pl.DataFrame) -> None:
    inferpipe = InferencePipeline(
        dataset=mock_dataset,
        predictor=Mock(spec=BasePredictor, predict=Mock(return_value=mock_results)),
    )
    out = inferpipe.process()
    assert_frame_equal(out, mock_results)


def test_inference_pipeline_process_with_path(
    mock_dataset: pl.DataFrame, mock_results: pl.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path.joinpath("data").joinpath("results.parquet")
    inferpipe = InferencePipeline(
        dataset=mock_dataset,
        predictor=Mock(spec=BasePredictor, predict=Mock(return_value=mock_results)),
        path=path,
    )
    out = inferpipe.process()
    assert_frame_equal(out, mock_results)

    assert path.is_file()
    assert_frame_equal(pl.read_parquet(path), mock_results)


def test_inference_pipeline_process_with_existing_results(
    mock_dataset: pl.DataFrame, mock_results: pl.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path.joinpath("results.parquet")
    pl.DataFrame({"name": ["a", "b", "c"]}).write_parquet(path)
    inferpipe = InferencePipeline(
        dataset=mock_dataset,
        predictor=Mock(spec=BasePredictor, predict=Mock(return_value=mock_results)),
        path=path,
    )
    out = inferpipe.process()
    assert_frame_equal(out, pl.DataFrame({"name": ["a", "b", "c"]}))

    assert path.is_file()
    assert_frame_equal(pl.read_parquet(path), pl.DataFrame({"name": ["a", "b", "c"]}))
