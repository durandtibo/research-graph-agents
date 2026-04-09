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


@pytest.fixture
def perfect_results() -> BinaryClassificationResults:
    return BinaryClassificationResults(
        n_samples=100,
        accuracy=1.0,
        true_positive=60,
        true_negative=40,
        false_positive=0,
        false_negative=0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        specificity=1.0,
    )


@pytest.fixture
def worst_results() -> BinaryClassificationResults:
    return BinaryClassificationResults(
        n_samples=100,
        accuracy=0.0,
        true_positive=0,
        true_negative=0,
        false_positive=40,
        false_negative=60,
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        specificity=0.0,
    )


@pytest.fixture
def typical_results() -> BinaryClassificationResults:
    return BinaryClassificationResults(
        n_samples=200,
        accuracy=0.85,
        true_positive=90,
        true_negative=80,
        false_positive=15,
        false_negative=15,
        precision=0.857,
        recall=0.857,
        f1_score=0.857,
        specificity=0.842,
    )


#################################################
#     Tests for BinaryClassificationResults     #
#################################################


def test_binary_classification_results_instantiation(
    typical_results: BinaryClassificationResults,
) -> None:
    assert typical_results.n_samples == 200
    assert typical_results.accuracy == 0.85
    assert typical_results.true_positive == 90
    assert typical_results.true_negative == 80
    assert typical_results.false_positive == 15
    assert typical_results.false_negative == 15
    assert typical_results.precision == 0.857
    assert typical_results.recall == 0.857
    assert typical_results.f1_score == 0.857
    assert typical_results.specificity == 0.842


def test_binary_classification_results_is_mutable() -> None:
    results = BinaryClassificationResults(
        n_samples=100,
        accuracy=0.9,
        true_positive=50,
        true_negative=40,
        false_positive=5,
        false_negative=5,
        precision=0.9,
        recall=0.9,
        f1_score=0.9,
        specificity=0.9,
    )
    results.accuracy = 0.95
    assert results.accuracy == 0.95


def test_binary_classification_results_equality() -> None:
    r1 = BinaryClassificationResults(
        n_samples=100,
        accuracy=0.9,
        true_positive=50,
        true_negative=40,
        false_positive=5,
        false_negative=5,
        precision=0.9,
        recall=0.9,
        f1_score=0.9,
        specificity=0.9,
    )
    r2 = BinaryClassificationResults(
        n_samples=100,
        accuracy=0.9,
        true_positive=50,
        true_negative=40,
        false_positive=5,
        false_negative=5,
        precision=0.9,
        recall=0.9,
        f1_score=0.9,
        specificity=0.9,
    )
    assert r1 == r2


def test_binary_classification_results_inequality(
    typical_results: BinaryClassificationResults, perfect_results: BinaryClassificationResults
) -> None:
    assert typical_results != perfect_results


def test_binary_classification_results_to_str_typical(
    typical_results: BinaryClassificationResults,
) -> None:
    assert typical_results.to_str() == (
        "Classification Results (n=200)\n"
        "------------------------------\n"
        "Accuracy    [█████████████████░░░]  0.8500  (170/200)\n"
        "Precision   [█████████████████░░░]  0.8570  (90/105)\n"
        "Recall      [█████████████████░░░]  0.8570  (90/105)\n"
        "Specificity [█████████████████░░░]  0.8420  (80/95)\n"
        "F1 Score    [█████████████████░░░]  0.8570\n"
        "\n"
        "Confusion Matrix: TP=90  TN=80  FP=15  FN=15"
    )


def test_binary_classification_results_to_str_perfect(
    perfect_results: BinaryClassificationResults,
) -> None:
    assert perfect_results.to_str() == (
        "Classification Results (n=100)\n"
        "------------------------------\n"
        "Accuracy    [████████████████████]  1.0000  (100/100)\n"
        "Precision   [████████████████████]  1.0000  (60/60)\n"
        "Recall      [████████████████████]  1.0000  (60/60)\n"
        "Specificity [████████████████████]  1.0000  (40/40)\n"
        "F1 Score    [████████████████████]  1.0000\n"
        "\n"
        "Confusion Matrix: TP=60  TN=40  FP=0  FN=0"
    )


def test_binary_classification_results_to_str_worst(
    worst_results: BinaryClassificationResults,
) -> None:
    assert worst_results.to_str() == (
        "Classification Results (n=100)\n"
        "------------------------------\n"
        "Accuracy    [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/100)\n"
        "Precision   [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/40)\n"
        "Recall      [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/60)\n"
        "Specificity [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/40)\n"
        "F1 Score    [░░░░░░░░░░░░░░░░░░░░]  0.0000\n"
        "\n"
        "Confusion Matrix: TP=0  TN=0  FP=40  FN=60"
    )


def test_binary_classification_results_to_str_single_sample() -> None:
    results = BinaryClassificationResults(
        n_samples=1,
        accuracy=1.0,
        true_positive=1,
        true_negative=0,
        false_positive=0,
        false_negative=0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        specificity=0.0,
    )
    assert results.to_str() == (
        "Classification Results (n=1)\n"
        "----------------------------\n"
        "Accuracy    [████████████████████]  1.0000  (1/1)\n"
        "Precision   [████████████████████]  1.0000  (1/1)\n"
        "Recall      [████████████████████]  1.0000  (1/1)\n"
        "Specificity [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/0)\n"
        "F1 Score    [████████████████████]  1.0000\n"
        "\n"
        "Confusion Matrix: TP=1  TN=0  FP=0  FN=0"
    )


###########################################################
#     Tests for compute_binary_classification_metrics     #
###########################################################


def test_compute_binary_classification_metrics_balanced(dataframe: pl.DataFrame) -> None:
    result = compute_binary_classification_metrics(dataframe, "target", "predicted")
    assert result == BinaryClassificationResults(
        n_samples=10,
        true_positive=3,
        true_negative=4,
        false_positive=1,
        false_negative=2,
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
        true_positive=2,
        true_negative=2,
        false_positive=0,
        false_negative=0,
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
        true_positive=0,
        true_negative=0,
        false_positive=2,
        false_negative=2,
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
        true_positive=1,
        true_negative=0,
        false_positive=3,
        false_negative=0,
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
        true_positive=1,
        true_negative=0,
        false_positive=0,
        false_negative=3,
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
        true_positive=2,
        true_negative=2,
        false_positive=0,
        false_negative=0,
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
        true_positive=0,
        true_negative=2,
        false_positive=0,
        false_negative=0,
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
