from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import polars as pl
import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import Runnable
from polars.testing import assert_frame_equal

from argos.autoprompt.haiku import columns
from argos.autoprompt.haiku.predictor import Predictor, generate_predictions
from argos.models.haiku_judge import HaikuJudgeResult

MODULE = "argos.autoprompt.haiku.predictor"


@pytest.fixture
def mock_dataset() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
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
            columns.TOPIC: "rain",
            columns.HAIKU: (
                "Gray sky descends slow,\n"
                "Cool drops kiss the thirsty ground,\n"
                "Silence finds the leaf."
            ),
            "evaluation": HaikuJudgeResult(
                structure_prediction=True,
                topic_prediction=True,
                score_prediction=10,
                score_reasoning="score explanation",
                structure_reasoning="structure explanation",
                topic_reasoning="topic explanation",
                overall_prediction=True,
            ),
        },
        {
            columns.TOPIC: "cat",
            columns.HAIKU: (
                "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
            ),
            "evaluation": HaikuJudgeResult(
                structure_prediction=True,
                topic_prediction=True,
                score_prediction=9,
                score_reasoning="score explanation",
                structure_reasoning="structure explanation",
                topic_reasoning="topic explanation",
                overall_prediction=True,
            ),
        },
        {
            columns.TOPIC: "mountain",
            columns.HAIKU: (
                "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
            ),
            "evaluation": HaikuJudgeResult(
                structure_prediction=True,
                topic_prediction=True,
                score_prediction=8,
                score_reasoning="score explanation",
                structure_reasoning="structure explanation",
                topic_reasoning="topic explanation",
                overall_prediction=True,
            ),
        },
    ]


@pytest.fixture
def mock_predictions() -> pl.DataFrame:
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
                columns.SCORE_PREDICTION: 10,
                columns.SCORE_REASONING: "score explanation",
                columns.STRUCTURE_PREDICTION: True,
                columns.STRUCTURE_REASONING: "structure explanation",
                columns.STRUCTURE_TARGET: True,
                columns.OVERALL_TARGET: True,
                columns.TOPIC_PREDICTION: True,
                columns.TOPIC_REASONING: "topic explanation",
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
                columns.SCORE_PREDICTION: 9,
                columns.SCORE_REASONING: "score explanation",
                columns.STRUCTURE_PREDICTION: True,
                columns.STRUCTURE_REASONING: "structure explanation",
                columns.STRUCTURE_TARGET: True,
                columns.OVERALL_TARGET: True,
                columns.TOPIC_PREDICTION: True,
                columns.TOPIC_REASONING: "topic explanation",
                columns.TOPIC_TARGET: True,
            },
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                columns.OVERALL_PREDICTION: True,
                columns.SCORE_PREDICTION: 8,
                columns.SCORE_REASONING: "score explanation",
                columns.STRUCTURE_PREDICTION: True,
                columns.STRUCTURE_REASONING: "structure explanation",
                columns.STRUCTURE_TARGET: True,
                columns.OVERALL_TARGET: True,
                columns.TOPIC_PREDICTION: True,
                columns.TOPIC_REASONING: "topic explanation",
                columns.TOPIC_TARGET: True,
            },
        ],
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
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=10,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
                    overall_prediction=True,
                ),
            }
        ],
        [
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=9,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
                    overall_prediction=True,
                ),
            },
        ],
        [
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=8,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
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
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=10,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
                    overall_prediction=True,
                ),
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=9,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
                    overall_prediction=True,
                ),
            },
        ],
        [
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=8,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
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
    predictor = Predictor(
        model=mock_model, output_columns=[columns.TOPIC, columns.HAIKU, columns.OVERALL_PREDICTION]
    )
    predictions = predictor.predict(mock_dataset)
    assert_frame_equal(
        predictions,
        pl.from_dicts(
            [
                {
                    columns.TOPIC: "rain",
                    columns.HAIKU: (
                        "Gray sky descends slow,\n"
                        "Cool drops kiss the thirsty ground,\n"
                        "Silence finds the leaf."
                    ),
                    columns.OVERALL_PREDICTION: True,
                },
                {
                    columns.TOPIC: "cat",
                    columns.HAIKU: (
                        "Soft fur, warm light gleam,\n"
                        "Silent paws upon the floor,\n"
                        "Sunbeam, peace descends."
                    ),
                    columns.OVERALL_PREDICTION: True,
                },
                {
                    columns.TOPIC: "mountain",
                    columns.HAIKU: (
                        "Snow upon the peak\n"
                        "Clouds are resting on the stone\n"
                        "Quiet, cold, and still"
                    ),
                    columns.OVERALL_PREDICTION: True,
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
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=10,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
                    overall_prediction=True,
                ),
            }
        ],
        [
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=9,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
                    overall_prediction=True,
                ),
            },
        ],
        [
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=8,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
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
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=10,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
                    overall_prediction=True,
                ),
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=9,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
                    overall_prediction=True,
                ),
            },
        ],
        [
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "evaluation": HaikuJudgeResult(
                    structure_prediction=True,
                    topic_prediction=True,
                    score_prediction=8,
                    overall_prediction=True,
                    score_reasoning="score explanation",
                    structure_reasoning="structure explanation",
                    topic_reasoning="topic explanation",
                ),
            },
        ],
    ]
    predictions = generate_predictions(dataset=mock_dataset, model=mock_model, batch_size=2)
    assert_frame_equal(predictions, mock_predictions, check_column_order=False)
