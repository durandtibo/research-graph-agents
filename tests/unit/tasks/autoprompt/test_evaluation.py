from __future__ import annotations

import polars as pl
import pytest

from argos.metrics import BinaryClassificationResults
from argos.tasks.autoprompt.evaluation import evaluate_metrics

MODULE = "argos.tasks.autoprompt.judge"


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


######################################
#     Tests for evaluate_metrics     #
######################################


def test_evaluate_metrics(mock_results: pl.DataFrame) -> None:
    assert evaluate_metrics(mock_results) == {
        "overall": BinaryClassificationResults(
            n_samples=3,
            accuracy=1.0,
            tp=3,
            tn=0,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            specificity=0.0,
        ),
        "structure": BinaryClassificationResults(
            n_samples=3,
            accuracy=1.0,
            tp=3,
            tn=0,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            specificity=0.0,
        ),
        "topic": BinaryClassificationResults(
            n_samples=3,
            accuracy=1.0,
            tp=3,
            tn=0,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            specificity=0.0,
        ),
    }


def test_evaluate_metrics_mixed_results() -> None:
    assert evaluate_metrics(
        pl.from_dicts(
            [
                {
                    "structure_target": True,
                    "topic_target": True,
                    "target": True,
                    "passed": False,
                    "structure_passed": True,
                    "topic_passed": True,
                },
                {
                    "structure_target": True,
                    "topic_target": True,
                    "target": True,
                    "passed": False,
                    "structure_passed": False,
                    "topic_passed": True,
                },
                {
                    "structure_target": True,
                    "topic_target": True,
                    "target": True,
                    "passed": False,
                    "structure_passed": False,
                    "topic_passed": False,
                },
                {
                    "structure_target": True,
                    "topic_target": True,
                    "target": True,
                    "passed": True,
                    "structure_passed": True,
                    "topic_passed": True,
                },
            ]
        )
    ) == {
        "overall": BinaryClassificationResults(
            n_samples=4,
            accuracy=0.25,
            tp=1,
            tn=0,
            fp=0,
            fn=3,
            precision=1.0,
            recall=0.25,
            f1_score=0.4,
            specificity=0.0,
        ),
        "structure": BinaryClassificationResults(
            n_samples=4,
            accuracy=0.5,
            tp=2,
            tn=0,
            fp=0,
            fn=2,
            precision=1.0,
            recall=0.5,
            f1_score=pytest.approx(2.0 / 3.0, abs=1e-6),
            specificity=0.0,
        ),
        "topic": BinaryClassificationResults(
            n_samples=4,
            accuracy=0.75,
            tp=3,
            tn=0,
            fp=0,
            fn=1,
            precision=1.0,
            recall=0.75,
            f1_score=pytest.approx(1.5 / 1.75, abs=1e-6),
            specificity=0.0,
        ),
    }
