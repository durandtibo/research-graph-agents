from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import polars as pl
import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import Runnable
from polars.testing import assert_frame_equal

from argos.autoprompt.haiku.predictor import Predictor, generate_predictions
from argos.models.haiku_judge import HaikuJudgeResult

MODULE = "argos.autoprompt.haiku.predictor"


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
def mock_llm() -> BaseChatModel:
    return Mock(spec=BaseChatModel, model="gpt-4o", temperature=0)


@pytest.fixture
def mock_model(mock_outputs: list[dict[str, Any]]) -> Runnable:
    model = Mock(spec=Runnable)
    model.batch.side_effect = [mock_outputs]
    return model


@pytest.fixture
def mock_outputs() -> list[dict[str, Any]]:
    return [
        {
            "topic": "rain",
            "haiku": (
                "Gray sky descends slow,\n"
                "Cool drops kiss the thirsty ground,\n"
                "Silence finds the leaf."
            ),
            "evaluation": HaikuJudgeResult(
                structure_prediction=True,
                topic_prediction=True,
                score=10,
                overall_reasoning="reason1",
                overall_prediction=True,
            ),
        },
        {
            "topic": "cat",
            "haiku": (
                "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
            ),
            "evaluation": HaikuJudgeResult(
                structure_prediction=True,
                topic_prediction=True,
                score=9,
                overall_reasoning="reason2",
                overall_prediction=True,
            ),
        },
        {
            "topic": "mountain",
            "haiku": (
                "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
            ),
            "evaluation": HaikuJudgeResult(
                structure_prediction=True,
                topic_prediction=True,
                score=8,
                overall_reasoning="reason3",
                overall_prediction=True,
            ),
        },
    ]


@pytest.fixture
def mock_predictions() -> pl.DataFrame:
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
                "overall_prediction": True,
                "target": True,
                "structure_prediction": True,
                "structure_reasoning": None,
                "structure_target": True,
                "topic_prediction": True,
                "topic_reasoning": None,
                "topic_target": True,
                "overall_reasoning": "reason1",
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                "score": 9,
                "overall_prediction": True,
                "target": True,
                "structure_prediction": True,
                "structure_reasoning": None,
                "structure_target": True,
                "topic_prediction": True,
                "topic_reasoning": None,
                "topic_target": True,
                "overall_reasoning": "reason2",
            },
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "score": 8,
                "overall_prediction": True,
                "target": True,
                "structure_prediction": True,
                "structure_reasoning": None,
                "structure_target": True,
                "topic_prediction": True,
                "topic_reasoning": None,
                "topic_target": True,
                "overall_reasoning": "reason3",
            },
        ]
    )


###############################
#     Tests for Predictor     #
###############################


def test_predictor_repr(mock_model: Runnable) -> None:
    assert repr(Predictor(model=mock_model)).startswith("Predictor(")


def test_predictor_str(mock_model: Runnable) -> None:
    assert str(Predictor(model=mock_model)).startswith("Predictor(")


def test_predictor_predict(
    mock_dataset: pl.DataFrame, mock_model: Runnable, mock_predictions: pl.DataFrame
) -> None:
    predictor = Predictor(model=mock_model)
    predictions = predictor.predict(mock_dataset)
    assert_frame_equal(predictions, mock_predictions, check_column_order=False)


def test_predictor_predict_batch_size_1(
    mock_dataset: pl.DataFrame, mock_predictions: pl.DataFrame
) -> None:
    mock_model = Mock(spec=Runnable)
    mock_model.batch.side_effect = [
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=10,
                    overall_reasoning="reason1",
                    overall_prediction=True,
                ),
            }
        ],
        [
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=9,
                    overall_reasoning="reason2",
                    overall_prediction=True,
                ),
            },
        ],
        [
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=8,
                    overall_reasoning="reason3",
                    overall_prediction=True,
                ),
            },
        ],
    ]
    predictor = Predictor(model=mock_model, batch_size=1)
    predictions = predictor.predict(mock_dataset)
    assert_frame_equal(predictions, mock_predictions, check_column_order=False)


def test_predictor_predict_batch_size_2(
    mock_dataset: pl.DataFrame, mock_predictions: pl.DataFrame
) -> None:
    mock_model = Mock(spec=Runnable)
    mock_model.batch.side_effect = [
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=10,
                    overall_reasoning="reason1",
                    overall_prediction=True,
                ),
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=9,
                    overall_reasoning="reason2",
                    overall_prediction=True,
                ),
            },
        ],
        [
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=8,
                    overall_reasoning="reason3",
                    overall_prediction=True,
                ),
            },
        ],
    ]
    predictor = Predictor(model=mock_model, batch_size=2)
    predictions = predictor.predict(mock_dataset)
    assert_frame_equal(predictions, mock_predictions, check_column_order=False)


def test_predictor_predict_selects_output_columns(
    mock_dataset: pl.DataFrame, mock_model: Runnable
) -> None:
    predictor = Predictor(model=mock_model, output_columns=["topic", "haiku", "overall_prediction"])
    predictions = predictor.predict(mock_dataset)
    assert_frame_equal(
        predictions,
        pl.from_dicts(
            [
                {
                    "topic": "rain",
                    "haiku": (
                        "Gray sky descends slow,\n"
                        "Cool drops kiss the thirsty ground,\n"
                        "Silence finds the leaf."
                    ),
                    "overall_prediction": True,
                },
                {
                    "topic": "cat",
                    "haiku": (
                        "Soft fur, warm light gleam,\n"
                        "Silent paws upon the floor,\n"
                        "Sunbeam, peace descends."
                    ),
                    "overall_prediction": True,
                },
                {
                    "topic": "mountain",
                    "haiku": (
                        "Snow upon the peak\n"
                        "Clouds are resting on the stone\n"
                        "Quiet, cold, and still"
                    ),
                    "overall_prediction": True,
                },
            ]
        ),
    )


##########################################
#     Tests for generate_predictions     #
##########################################


def test_generate_predictions(
    mock_dataset: pl.DataFrame, mock_model: Runnable, mock_predictions: pl.DataFrame
) -> None:
    predictions = generate_predictions(dataset=mock_dataset, model=mock_model)
    assert_frame_equal(predictions, mock_predictions, check_column_order=False)


def test_generate_predictions_batch_size_1(
    mock_dataset: pl.DataFrame, mock_predictions: pl.DataFrame
) -> None:
    mock_model = Mock(spec=Runnable)
    mock_model.batch.side_effect = [
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=10,
                    overall_reasoning="reason1",
                    overall_prediction=True,
                ),
            }
        ],
        [
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=9,
                    overall_reasoning="reason2",
                    overall_prediction=True,
                ),
            },
        ],
        [
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=8,
                    overall_reasoning="reason3",
                    overall_prediction=True,
                ),
            },
        ],
    ]
    predictions = generate_predictions(dataset=mock_dataset, model=mock_model, batch_size=1)
    assert_frame_equal(predictions, mock_predictions, check_column_order=False)


def test_generate_predictions_batch_size_2(
    mock_dataset: pl.DataFrame, mock_predictions: pl.DataFrame
) -> None:
    mock_model = Mock(spec=Runnable)
    mock_model.batch.side_effect = [
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=10,
                    overall_reasoning="reason1",
                    overall_prediction=True,
                ),
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=9,
                    overall_reasoning="reason2",
                    overall_prediction=True,
                ),
            },
        ],
        [
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score=8,
                    overall_reasoning="reason3",
                    overall_prediction=True,
                ),
            },
        ],
    ]
    predictions = generate_predictions(dataset=mock_dataset, model=mock_model, batch_size=2)
    assert_frame_equal(predictions, mock_predictions, check_column_order=False)
