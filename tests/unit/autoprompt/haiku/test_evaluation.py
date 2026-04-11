from __future__ import annotations

import polars as pl
import pytest

from argos.autoprompt.haiku import columns
from argos.autoprompt.haiku.evaluation import evaluate_classification_metrics
from argos.metrics import BinaryClassificationResults


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
def correct_metrics() -> dict[str, BinaryClassificationResults]:
    return {
        "overall": BinaryClassificationResults(
            n_samples=3,
            accuracy=1.0,
            true_positive=3,
            true_negative=0,
            false_positive=0,
            false_negative=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            specificity=0.0,
        ),
        "structure": BinaryClassificationResults(
            n_samples=3,
            accuracy=1.0,
            true_positive=3,
            true_negative=0,
            false_positive=0,
            false_negative=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            specificity=0.0,
        ),
        "topic": BinaryClassificationResults(
            n_samples=3,
            accuracy=1.0,
            true_positive=3,
            true_negative=0,
            false_positive=0,
            false_negative=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            specificity=0.0,
        ),
    }


@pytest.fixture
def mixed_metrics() -> dict[str, BinaryClassificationResults]:
    return {
        "overall": BinaryClassificationResults(
            n_samples=4,
            accuracy=0.25,
            true_positive=1,
            true_negative=0,
            false_positive=0,
            false_negative=3,
            precision=1.0,
            recall=0.25,
            f1_score=0.4,
            specificity=0.0,
        ),
        "structure": BinaryClassificationResults(
            n_samples=4,
            accuracy=0.5,
            true_positive=2,
            true_negative=0,
            false_positive=0,
            false_negative=2,
            precision=1.0,
            recall=0.5,
            f1_score=pytest.approx(2.0 / 3.0, abs=1e-6),
            specificity=0.0,
        ),
        "topic": BinaryClassificationResults(
            n_samples=4,
            accuracy=0.75,
            true_positive=3,
            true_negative=0,
            false_positive=0,
            false_negative=1,
            precision=1.0,
            recall=0.75,
            f1_score=pytest.approx(1.5 / 1.75, abs=1e-6),
            specificity=0.0,
        ),
    }


#####################################################
#     Tests for evaluate_classification_metrics     #
#####################################################


def test_evaluate_classification_metrics(
    correct_predictions: pl.DataFrame, correct_metrics: dict[str, BinaryClassificationResults]
) -> None:
    assert evaluate_classification_metrics(correct_predictions) == correct_metrics


def test_evaluate_classification_metrics_mixed_results(
    mixed_predictions: pl.DataFrame, mixed_metrics: dict[str, BinaryClassificationResults]
) -> None:
    assert evaluate_classification_metrics(mixed_predictions) == mixed_metrics
