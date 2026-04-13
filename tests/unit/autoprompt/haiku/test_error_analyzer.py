from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import polars as pl
import pytest
from iden.io import load_text
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from argos.autoprompt.haiku import columns
from argos.autoprompt.haiku.error_analyzer import ErrorAnalyzer
from argos.autoprompt.haiku.error_finder import BaseErrorFinder

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def mock_error_finder() -> BaseErrorFinder:
    return Mock(spec=BaseErrorFinder, find=Mock(return_value="a list of errors blabla..."))


@pytest.fixture
def mock_model() -> Runnable:
    return Mock(spec=Runnable, invoke=Mock(side_effect=[AIMessage("analysis of the errors")]))


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


###################################
#     Tests for ErrorAnalyzer     #
###################################


def test_error_analyzer_repr(mock_error_finder: BaseErrorFinder, mock_model: Runnable) -> None:
    assert repr(ErrorAnalyzer(error_finder=mock_error_finder, model=mock_model)).startswith(
        "ErrorAnalyzer("
    )


def test_error_analyzer_str(mock_error_finder: BaseErrorFinder, mock_model: Runnable) -> None:
    assert str(ErrorAnalyzer(error_finder=mock_error_finder, model=mock_model)).startswith(
        "ErrorAnalyzer("
    )


def test_error_analyzer_analyze(
    mock_error_finder: BaseErrorFinder, mock_model: Runnable, correct_predictions: pl.DataFrame
) -> None:
    assert (
        ErrorAnalyzer(error_finder=mock_error_finder, model=mock_model).analyze(correct_predictions)
        == "analysis of the errors"
    )
    mock_error_finder.find.assert_called_once_with(correct_predictions)
    mock_model.invoke.assert_called_once_with({"text": "a list of errors blabla..."})


def test_error_analyzer_analyze_with_path(
    mock_error_finder: BaseErrorFinder,
    mock_model: Runnable,
    correct_predictions: pl.DataFrame,
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath("data").joinpath("errors.md")
    assert (
        ErrorAnalyzer(error_finder=mock_error_finder, model=mock_model, path=path).analyze(
            correct_predictions
        )
        == "analysis of the errors"
    )
    mock_error_finder.find.assert_called_once_with(correct_predictions)
    mock_model.invoke.assert_called_once_with({"text": "a list of errors blabla..."})
    assert path.is_file()
    assert load_text(path) == "analysis of the errors"


def test_error_analyzer_analyze_model_outputs_aimessage(
    mock_error_finder: BaseErrorFinder, mock_model: Runnable, correct_predictions: pl.DataFrame
) -> None:
    assert (
        ErrorAnalyzer(error_finder=mock_error_finder, model=mock_model).analyze(correct_predictions)
        == "analysis of the errors"
    )
    mock_error_finder.find.assert_called_once_with(correct_predictions)
    mock_model.invoke.assert_called_once_with({"text": "a list of errors blabla..."})


def test_error_analyzer_analyze_model_outputs_dict(
    mock_error_finder: BaseErrorFinder, correct_predictions: pl.DataFrame
) -> None:
    mock_model = Mock(
        spec=Runnable, invoke=Mock(side_effect=[{"analysis": "analysis of the errors"}])
    )
    assert (
        ErrorAnalyzer(error_finder=mock_error_finder, model=mock_model).analyze(correct_predictions)
        == "analysis of the errors"
    )
    mock_error_finder.find.assert_called_once_with(correct_predictions)
    mock_model.invoke.assert_called_once_with({"text": "a list of errors blabla..."})
