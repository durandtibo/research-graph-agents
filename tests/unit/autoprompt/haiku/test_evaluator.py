from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from coola.equality import objects_are_allclose, objects_are_equal
from iden.io import load_json

from argos.autoprompt.haiku import columns
from argos.autoprompt.haiku.evaluator import HaikuJudgeEvaluator

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
                columns.OVERALL_SCORE_PREDICTION: 10,
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
                columns.OVERALL_SCORE_PREDICTION: 9,
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
                columns.OVERALL_SCORE_PREDICTION: 8,
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
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
                columns.OVERALL_PREDICTION: False,
                columns.STRUCTURE_PREDICTION: True,
                columns.TOPIC_PREDICTION: True,
            },
            {
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
                columns.OVERALL_PREDICTION: False,
                columns.STRUCTURE_PREDICTION: False,
                columns.TOPIC_PREDICTION: True,
            },
            {
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
                columns.OVERALL_PREDICTION: False,
                columns.STRUCTURE_PREDICTION: False,
                columns.TOPIC_PREDICTION: False,
            },
            {
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
                columns.OVERALL_PREDICTION: True,
                columns.STRUCTURE_PREDICTION: True,
                columns.TOPIC_PREDICTION: True,
            },
        ]
    )


@pytest.fixture
def correct_metrics() -> dict[str, Any]:
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


@pytest.fixture
def mixed_metrics() -> dict[str, Any]:
    return {
        "overall": {
            "n_samples": 4,
            "accuracy": 0.25,
            "true_positive": 1,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 3,
            "precision": 1.0,
            "recall": 0.25,
            "f1_score": 0.4,
            "specificity": 0.0,
        },
        "structure": {
            "n_samples": 4,
            "accuracy": 0.5,
            "true_positive": 2,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 2,
            "precision": 1.0,
            "recall": 0.5,
            "f1_score": 2.0 / 3.0,
            "specificity": 0.0,
        },
        "topic": {
            "n_samples": 4,
            "accuracy": 0.75,
            "true_positive": 3,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 1,
            "precision": 1.0,
            "recall": 0.75,
            "f1_score": 1.5 / 1.75,
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


def test_haiku_judge_evaluator_evaluate_correct(
    correct_predictions: pl.DataFrame, correct_metrics: dict[str, Any]
) -> None:
    evaluator = HaikuJudgeEvaluator()
    metrics = evaluator.evaluate(correct_predictions)
    assert objects_are_equal(metrics, correct_metrics)


def test_haiku_judge_evaluator_evaluate_mixed(
    mixed_predictions: pl.DataFrame, mixed_metrics: dict[str, Any]
) -> None:
    evaluator = HaikuJudgeEvaluator()
    metrics = evaluator.evaluate(mixed_predictions)
    assert objects_are_allclose(metrics, mixed_metrics, atol=1e-6)


def test_haiku_judge_evaluator_evaluate_with_path(
    correct_predictions: pl.DataFrame, correct_metrics: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path.joinpath("data").joinpath("metrics.json")
    evaluator = HaikuJudgeEvaluator(path)
    metrics = evaluator.evaluate(correct_predictions)
    assert objects_are_equal(metrics, correct_metrics)
    assert path.is_file()
    assert objects_are_equal(load_json(path), correct_metrics)
