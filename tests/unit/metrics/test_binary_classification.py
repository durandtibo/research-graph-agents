from __future__ import annotations

from dataclasses import asdict

import polars as pl
import pytest

from argos.metrics import (
    BinaryClassificationResults,
    compute_binary_classification_metrics,
)


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {"target": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0], "prediction": [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]}
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


def test_binary_classification_results_allclose_true() -> None:
    assert BinaryClassificationResults(
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
    ).allclose(
        BinaryClassificationResults(
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
    )


def test_binary_classification_results_allclose_true_atol() -> None:
    assert BinaryClassificationResults(
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
    ).allclose(
        BinaryClassificationResults(
            n_samples=100,
            accuracy=0.9,
            true_positive=50,
            true_negative=40,
            false_positive=5,
            false_negative=5,
            precision=0.9,
            recall=0.9,
            f1_score=0.9,
            specificity=0.9009,
        ),
        atol=1e-3,
    )


def test_binary_classification_results_allclose_true_rtol() -> None:
    assert BinaryClassificationResults(
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
    ).allclose(
        BinaryClassificationResults(
            n_samples=100,
            accuracy=0.9,
            true_positive=50,
            true_negative=40,
            false_positive=5,
            false_negative=5,
            precision=0.9,
            recall=0.9,
            f1_score=0.9,
            specificity=0.9009,
        ),
        rtol=1e-3,
    )


def test_binary_classification_results_allclose_true_same_object() -> None:
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
    assert results.allclose(results)


def test_binary_classification_results_allclose_false_different_value() -> None:
    assert not BinaryClassificationResults(
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
    ).allclose(
        BinaryClassificationResults(
            n_samples=101,
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
    )


def test_binary_classification_results_allclose_false_different_type() -> None:
    assert not BinaryClassificationResults(
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
    ).allclose(42)


def test_binary_classification_results_allclose_false_different_type_child() -> None:
    class MyChild(BinaryClassificationResults): ...

    assert not BinaryClassificationResults(
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
    ).allclose(
        MyChild(
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
    )


def test_binary_classification_results_equal_true() -> None:
    assert BinaryClassificationResults(
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
    ).equal(
        BinaryClassificationResults(
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
    )


def test_binary_classification_results_equal_true_same_object() -> None:
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
    assert results.equal(results)


def test_binary_classification_results_equal_false_different_value() -> None:
    assert not BinaryClassificationResults(
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
    ).equal(
        BinaryClassificationResults(
            n_samples=101,
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
    )


def test_binary_classification_results_equal_false_different_type() -> None:
    assert not BinaryClassificationResults(
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
    ).equal(42)


def test_binary_classification_results_equal_false_different_type_child() -> None:
    class MyChild(BinaryClassificationResults): ...

    assert not BinaryClassificationResults(
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
    ).equal(
        MyChild(
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
    )


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
    result = compute_binary_classification_metrics(
        dataframe, target_col="target", prediction_col="prediction"
    )
    assert result.allclose(
        BinaryClassificationResults(
            n_samples=10,
            true_positive=3,
            true_negative=4,
            false_positive=1,
            false_negative=2,
            accuracy=0.7,
            precision=0.75,
            recall=0.6,
            f1_score=2 / 3,
            specificity=0.8,
        ),
        atol=1e-6,
    )


def test_compute_binary_classification_metrics_perfect_predictions() -> None:
    df = pl.DataFrame({"target": [1, 0, 1, 0], "prediction": [1, 0, 1, 0]})
    result = compute_binary_classification_metrics(
        df, target_col="target", prediction_col="prediction"
    )
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
    df = pl.DataFrame({"target": [1, 0, 1, 0], "prediction": [0, 1, 0, 1]})
    result = compute_binary_classification_metrics(
        df, target_col="target", prediction_col="prediction"
    )
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
    df = pl.DataFrame({"target": [1, 0, 0, 0], "prediction": [1, 1, 1, 1]})
    result = compute_binary_classification_metrics(
        df, target_col="target", prediction_col="prediction"
    )
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
    df = pl.DataFrame({"target": [1, 1, 1, 1], "prediction": [1, 0, 0, 0]})
    result = compute_binary_classification_metrics(
        df, target_col="target", prediction_col="prediction"
    )
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
        {"target": [True, False, True, False], "prediction": [True, False, True, False]}
    )
    result = compute_binary_classification_metrics(
        df, target_col="target", prediction_col="prediction"
    )
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
    df = pl.DataFrame({"target": [0, 0], "prediction": [0, 0]})
    result = compute_binary_classification_metrics(
        df, target_col="target", prediction_col="prediction"
    )
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
    df = pl.DataFrame({"target": [], "prediction": []})
    with pytest.raises(ValueError, match="DataFrame is empty"):
        compute_binary_classification_metrics(df, target_col="target", prediction_col="prediction")


def test_compute_binary_classification_metrics_raises_on_non_binary_values() -> None:
    df = pl.DataFrame({"target": [1, 2, 3], "prediction": [1, 0, 1]})
    with pytest.raises(ValueError, match="non-binary values"):
        compute_binary_classification_metrics(df, target_col="target", prediction_col="prediction")


def test_compute_binary_classification_metrics_raises_on_non_binary_prediction_col() -> None:
    df = pl.DataFrame({"target": [1, 0, 1], "prediction": [1, 0, 2]})
    with pytest.raises(ValueError, match="non-binary values"):
        compute_binary_classification_metrics(df, target_col="target", prediction_col="prediction")


def test_compute_binary_classification_metrics_custom_column_names() -> None:
    df = pl.DataFrame({"t": [1, 0, 1, 0], "p": [1, 0, 1, 0]})
    result = compute_binary_classification_metrics(df, target_col="t", prediction_col="p")
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


@pytest.mark.parametrize(
    ("target", "prediction", "expected"),
    [
        pytest.param(
            [1],
            [1],
            BinaryClassificationResults(
                n_samples=1,
                true_positive=1,
                true_negative=0,
                false_positive=0,
                false_negative=0,
                accuracy=1.0,
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                specificity=0.0,
            ),
            id="true_positive",
        ),
        pytest.param(
            [0],
            [0],
            BinaryClassificationResults(
                n_samples=1,
                true_positive=0,
                true_negative=1,
                false_positive=0,
                false_negative=0,
                accuracy=1.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                specificity=1.0,
            ),
            id="true_negative",
        ),
        pytest.param(
            [1],
            [0],
            BinaryClassificationResults(
                n_samples=1,
                true_positive=0,
                true_negative=0,
                false_positive=0,
                false_negative=1,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                specificity=0.0,
            ),
            id="false_negative",
        ),
        pytest.param(
            [0],
            [1],
            BinaryClassificationResults(
                n_samples=1,
                true_positive=0,
                true_negative=0,
                false_positive=1,
                false_negative=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                specificity=0.0,
            ),
            id="false_positive",
        ),
    ],
)
def test_compute_binary_classification_metrics_single_sample(
    target: list[int],
    prediction: list[int],
    expected: BinaryClassificationResults,
) -> None:
    df = pl.DataFrame({"target": target, "prediction": prediction})
    result = compute_binary_classification_metrics(
        df, target_col="target", prediction_col="prediction"
    )
    assert result == expected


def test_compute_binary_classification_metrics_all_positives_target() -> None:
    df = pl.DataFrame({"target": [1, 1, 1], "prediction": [1, 1, 1]})
    result = compute_binary_classification_metrics(
        df, target_col="target", prediction_col="prediction"
    )
    assert result == BinaryClassificationResults(
        n_samples=3,
        true_positive=3,
        true_negative=0,
        false_positive=0,
        false_negative=0,
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        specificity=0.0,
    )


def test_binary_classification_results_asdict() -> None:
    results = BinaryClassificationResults(
        n_samples=10,
        accuracy=0.8,
        true_positive=5,
        true_negative=3,
        false_positive=1,
        false_negative=1,
        precision=0.833,
        recall=0.833,
        f1_score=0.833,
        specificity=0.75,
    )
    assert asdict(results) == {
        "n_samples": 10,
        "accuracy": 0.8,
        "true_positive": 5,
        "true_negative": 3,
        "false_positive": 1,
        "false_negative": 1,
        "precision": 0.833,
        "recall": 0.833,
        "f1_score": 0.833,
        "specificity": 0.75,
    }
