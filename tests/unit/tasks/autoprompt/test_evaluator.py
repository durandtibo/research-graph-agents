from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from coola.equality import objects_are_equal
from iden.io import load_json

from argos.tasks.autoprompt.evaluator import HaikuJudgeEvaluator

if TYPE_CHECKING:
    from pathlib import Path


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


@pytest.fixture
def mock_metrics() -> dict[str, Any]:
    return {
        "overall": {
            "n_samples": 3,
            "accuracy": 1.0,
            "true_positive": 3,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "specificity": 0.0,
        },
        "structure": {
            "n_samples": 3,
            "accuracy": 1.0,
            "true_positive": 3,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "specificity": 0.0,
        },
        "topic": {
            "n_samples": 3,
            "accuracy": 1.0,
            "true_positive": 3,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "specificity": 0.0,
        },
    }


#########################################
#     Tests for HaikuJudgeEvaluator     #
#########################################


def test_haiku_judge_evaluator_repr() -> None:
    assert repr(HaikuJudgeEvaluator()).startswith("HaikuJudgeEvaluator(")


def test_haiku_judge_evaluator_str() -> None:
    assert str(HaikuJudgeEvaluator()).startswith("HaikuJudgeEvaluator(")


def test_haiku_judge_evaluator_evaluate(
    mock_predictions: pl.DataFrame, mock_metrics: dict[str, Any]
) -> None:
    evaluator = HaikuJudgeEvaluator()
    metrics = evaluator.evaluate(mock_predictions)
    assert objects_are_equal(metrics, mock_metrics)


def test_haiku_judge_evaluator_evaluate_with_path(
    mock_predictions: pl.DataFrame, mock_metrics: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path.joinpath("data").joinpath("metrics.json")
    evaluator = HaikuJudgeEvaluator(path)
    metrics = evaluator.evaluate(mock_predictions)
    assert objects_are_equal(metrics, mock_metrics)
    assert path.is_file()
    assert objects_are_equal(load_json(path), mock_metrics)
