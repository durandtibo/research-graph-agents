from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import polars as pl
import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from polars.testing import assert_frame_equal

from argos.nodes.haiku_judge import HaikuJudgeResult
from argos.tasks.autoprompt.predictor import (
    Predictor,
    generate_predictions,
    prepare_results,
)

MODULE = "argos.tasks.autoprompt.judge"


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
    llm = Mock(spec=BaseChatModel)
    llm.model = "gpt-4o"
    llm.temperature = 0
    return llm


@pytest.fixture
def mock_graph(mock_outputs: list[dict[str, Any]]) -> CompiledStateGraph:
    graph = Mock(spec=CompiledStateGraph)
    graph.batch.side_effect = [mock_outputs]
    return graph


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
                structure_passed=True, topic_passed=True, score=10, reasoning="reason1", passed=True
            ),
        },
        {
            "topic": "cat",
            "haiku": (
                "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
            ),
            "evaluation": HaikuJudgeResult(
                structure_passed=True, topic_passed=True, score=9, reasoning="reason2", passed=True
            ),
        },
        {
            "topic": "mountain",
            "haiku": (
                "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
            ),
            "evaluation": HaikuJudgeResult(
                structure_passed=True, topic_passed=True, score=8, reasoning="reason3", passed=True
            ),
        },
    ]


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


###############################
#     Tests for Predictor     #
###############################


def test_predictor_predict(
    mock_dataset: pl.DataFrame, mock_graph: CompiledStateGraph, mock_results: pl.DataFrame
) -> None:
    predictor = Predictor(graph=mock_graph)
    results = predictor.predict(mock_dataset)
    assert_frame_equal(results, mock_results)


def test_predictor_predict_batch_size_1(
    mock_dataset: pl.DataFrame, mock_results: pl.DataFrame
) -> None:
    mock_graph = Mock(spec=CompiledStateGraph)
    mock_graph.batch.side_effect = [
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=10,
                    reasoning="reason1",
                    passed=True,
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
                    structure_passed=True,
                    topic_passed=True,
                    score=9,
                    reasoning="reason2",
                    passed=True,
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
                    structure_passed=True,
                    topic_passed=True,
                    score=8,
                    reasoning="reason3",
                    passed=True,
                ),
            },
        ],
    ]
    predictor = Predictor(graph=mock_graph, batch_size=1)
    results = predictor.predict(mock_dataset)
    assert_frame_equal(results, mock_results)


def test_predictor_predict_batch_size_2(
    mock_dataset: pl.DataFrame, mock_results: pl.DataFrame
) -> None:
    mock_graph = Mock(spec=CompiledStateGraph)
    mock_graph.batch.side_effect = [
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=10,
                    reasoning="reason1",
                    passed=True,
                ),
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=9,
                    reasoning="reason2",
                    passed=True,
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
                    structure_passed=True,
                    topic_passed=True,
                    score=8,
                    reasoning="reason3",
                    passed=True,
                ),
            },
        ],
    ]
    predictor = Predictor(graph=mock_graph, batch_size=2)
    results = predictor.predict(mock_dataset)
    assert_frame_equal(results, mock_results)


############################################
#     Tests for generate_predictions     #
############################################


def test_generate_predictions(
    mock_dataset: pl.DataFrame, mock_graph: CompiledStateGraph, mock_results: pl.DataFrame
) -> None:
    results = generate_predictions(dataset=mock_dataset, graph=mock_graph)
    assert_frame_equal(results, mock_results)


def test_generate_predictions_batch_size_1(
    mock_dataset: pl.DataFrame, mock_results: pl.DataFrame
) -> None:
    mock_graph = Mock(spec=CompiledStateGraph)
    mock_graph.batch.side_effect = [
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=10,
                    reasoning="reason1",
                    passed=True,
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
                    structure_passed=True,
                    topic_passed=True,
                    score=9,
                    reasoning="reason2",
                    passed=True,
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
                    structure_passed=True,
                    topic_passed=True,
                    score=8,
                    reasoning="reason3",
                    passed=True,
                ),
            },
        ],
    ]
    results = generate_predictions(dataset=mock_dataset, graph=mock_graph, batch_size=1)
    assert_frame_equal(results, mock_results)


def test_generate_predictions_batch_size_2(
    mock_dataset: pl.DataFrame, mock_results: pl.DataFrame
) -> None:
    mock_graph = Mock(spec=CompiledStateGraph)
    mock_graph.batch.side_effect = [
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=10,
                    reasoning="reason1",
                    passed=True,
                ),
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=9,
                    reasoning="reason2",
                    passed=True,
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
                    structure_passed=True,
                    topic_passed=True,
                    score=8,
                    reasoning="reason3",
                    passed=True,
                ),
            },
        ],
    ]
    results = generate_predictions(dataset=mock_dataset, graph=mock_graph, batch_size=2)
    assert_frame_equal(results, mock_results)


#####################################
#     Tests for prepare_results     #
#####################################


def test_prepare_results_returns_dataframe(
    mock_dataset: pl.DataFrame, mock_outputs: list, mock_results: pl.DataFrame
) -> None:
    assert_frame_equal(prepare_results(dataset=mock_dataset, outputs=mock_outputs), mock_results)
