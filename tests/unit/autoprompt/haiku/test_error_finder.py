from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from argos.autoprompt.haiku import columns
from argos.autoprompt.haiku.error_finder import ErrorFinder

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def correct_predictions() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                columns.OVERALL_PREDICTION: True,
                columns.OVERALL_TARGET: True,
                columns.STRUCTURE_PREDICTION: True,
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_PREDICTION: True,
                columns.TOPIC_TARGET: True,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                columns.OVERALL_PREDICTION: True,
                columns.OVERALL_TARGET: True,
                columns.STRUCTURE_PREDICTION: True,
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_PREDICTION: True,
                columns.TOPIC_TARGET: True,
            },
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                columns.OVERALL_PREDICTION: True,
                columns.OVERALL_TARGET: True,
                columns.STRUCTURE_PREDICTION: True,
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_PREDICTION: True,
                columns.TOPIC_TARGET: True,
            },
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                columns.OVERALL_PREDICTION: True,
                columns.OVERALL_TARGET: True,
                columns.STRUCTURE_PREDICTION: True,
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_PREDICTION: True,
                columns.TOPIC_TARGET: True,
            },
        ]
    )


@pytest.fixture
def mixed_predictions() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Gray and blue sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                columns.OVERALL_PREDICTION: True,
                columns.OVERALL_TARGET: False,
                columns.STRUCTURE_PREDICTION: True,
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_PREDICTION: True,
                columns.TOPIC_TARGET: True,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Soft fur and purr, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                columns.OVERALL_PREDICTION: True,
                columns.OVERALL_TARGET: False,
                columns.STRUCTURE_PREDICTION: True,
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_PREDICTION: True,
                columns.TOPIC_TARGET: True,
            },
            {
                columns.TOPIC: "snake",
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                columns.OVERALL_PREDICTION: True,
                columns.OVERALL_TARGET: False,
                columns.STRUCTURE_PREDICTION: True,
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_PREDICTION: True,
                columns.TOPIC_TARGET: False,
            },
            {
                columns.TOPIC: "coffee",
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                columns.OVERALL_PREDICTION: True,
                columns.OVERALL_TARGET: False,
                columns.STRUCTURE_PREDICTION: True,
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_PREDICTION: True,
                columns.TOPIC_TARGET: False,
            },
        ]
    )


#################################
#     Tests for ErrorFinder     #
#################################


def test_error_finder_repr() -> None:
    assert repr(ErrorFinder()).startswith("ErrorFinder(")


def test_error_finder_str() -> None:
    assert str(ErrorFinder()).startswith("ErrorFinder(")


def test_error_finder_find_correct(correct_predictions: pl.DataFrame) -> None:
    out = ErrorFinder().find(correct_predictions)
    assert "## Examples with Structure Errors" in out
    assert "0 haikus have incorrect structure predictions." in out
    assert "## Examples with Topic Errors" in out
    assert "0 haikus have incorrect topic predictions." in out
    assert "| # | Topic | Haiku | Target | Prediction |" in out


def test_error_finder_find_mixed(mixed_predictions: pl.DataFrame) -> None:
    out = ErrorFinder().find(mixed_predictions)
    assert "## Examples with Structure Errors" in out
    assert "2 haikus have incorrect structure predictions." in out
    assert "## Examples with Topic Errors" in out
    assert "2 haikus have incorrect topic predictions." in out
    assert "| # | Topic | Haiku | Target | Prediction |" in out


def test_error_finder_find_with_path(correct_predictions: pl.DataFrame, tmp_path: Path) -> None:
    finder = ErrorFinder(path=tmp_path.joinpath("data"))
    out = finder.find(correct_predictions)
    assert "## Examples with Structure Errors" in out
    assert "0 haikus have incorrect structure predictions." in out
    assert "## Examples with Topic Errors" in out
    assert "0 haikus have incorrect topic predictions." in out
    assert "| # | Topic | Haiku | Target | Prediction |" in out
    assert finder.structure_error_path.is_file()
    assert finder.topic_error_path.is_file()
