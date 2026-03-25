from __future__ import annotations

import polars as pl
import pytest

from argos.metrics import (
    BinaryClassificationResults,
    compute_binary_classification_metrics,
)


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {"target": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0], "predicted": [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]}
    )


###########################################################
#     Tests for compute_binary_classification_metrics     #
###########################################################


def test_compute_binary_classification_metrics_balanced(dataframe: pl.DataFrame) -> None:
    result = compute_binary_classification_metrics(dataframe, "target", "predicted")
    assert result == BinaryClassificationResults(
        n_samples=10,
        tp=3,
        tn=4,
        fp=1,
        fn=2,
        accuracy=0.7,
        precision=0.75,
        recall=0.6,
        f1_score=pytest.approx(0.6666, abs=1e-3),
        specificity=0.8,
    )


def test_compute_binary_classification_metrics_perfect_predictions() -> None:
    df = pl.DataFrame({"target": [1, 0, 1, 0], "predicted": [1, 0, 1, 0]})
    result = compute_binary_classification_metrics(df, "target", "predicted")
    assert result == BinaryClassificationResults(
        n_samples=4,
        tp=2,
        tn=2,
        fp=0,
        fn=0,
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        specificity=1.0,
    )


def test_compute_binary_classification_metrics_all_wrong_predictions() -> None:
    df = pl.DataFrame({"target": [1, 0, 1, 0], "predicted": [0, 1, 0, 1]})
    result = compute_binary_classification_metrics(df, "target", "predicted")
    assert result == BinaryClassificationResults(
        n_samples=4,
        tp=0,
        tn=0,
        fp=2,
        fn=2,
        accuracy=0.0,
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        specificity=0.0,
    )


def test_compute_binary_classification_metrics_low_precision() -> None:
    df = pl.DataFrame({"target": [1, 0, 0, 0], "predicted": [1, 1, 1, 1]})
    result = compute_binary_classification_metrics(df, "target", "predicted")
    assert result == BinaryClassificationResults(
        n_samples=4,
        tp=1,
        tn=0,
        fp=3,
        fn=0,
        accuracy=0.25,
        precision=0.25,
        recall=1.0,
        f1_score=0.4,
        specificity=0.0,
    )


def test_compute_binary_classification_metrics_low_recall() -> None:
    df = pl.DataFrame({"target": [1, 1, 1, 1], "predicted": [1, 0, 0, 0]})
    result = compute_binary_classification_metrics(df, "target", "predicted")
    assert result == BinaryClassificationResults(
        n_samples=4,
        tp=1,
        tn=0,
        fp=0,
        fn=3,
        accuracy=0.25,
        precision=1.0,
        recall=0.25,
        f1_score=0.4,
        specificity=0.0,
    )


def test_compute_binary_classification_metrics_with_boolean_values() -> None:
    df = pl.DataFrame(
        {"target": [True, False, True, False], "predicted": [True, False, True, False]}
    )
    result = compute_binary_classification_metrics(df, "target", "predicted")
    assert result == BinaryClassificationResults(
        n_samples=4,
        tp=2,
        tn=2,
        fp=0,
        fn=0,
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        specificity=1.0,
    )


def test_compute_binary_classification_metrics_zero_division() -> None:
    df = pl.DataFrame({"target": [0, 0], "predicted": [0, 0]})
    result = compute_binary_classification_metrics(df, "target", "predicted")
    assert result == BinaryClassificationResults(
        n_samples=2,
        tp=0,
        tn=2,
        fp=0,
        fn=0,
        accuracy=1.0,
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        specificity=1.0,
    )


def test_compute_binary_classification_metrics_raises_on_empty_dataframe() -> None:
    df = pl.DataFrame({"target": [], "predicted": []})
    with pytest.raises(ValueError, match="DataFrame is empty"):
        compute_binary_classification_metrics(df, "target", "predicted")


def test_compute_binary_classification_metrics_raises_on_non_binary_values() -> None:
    df = pl.DataFrame({"target": [1, 2, 3], "predicted": [1, 0, 1]})
    with pytest.raises(ValueError, match="non-binary values"):
        compute_binary_classification_metrics(df, "target", "predicted")
