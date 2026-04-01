r"""Contain binary classification metrics."""

from __future__ import annotations

__all__ = ["BinaryClassificationResults", "compute_binary_classification_metrics"]

from dataclasses import dataclass

import polars as pl
from coola.utils.format import make_bar


@dataclass
class BinaryClassificationResults:
    """Store classification metrics.

    Attributes:
        n_samples: The total number of samples in the DataFrame.
        accuracy: The proportion of correct predictions.
        tp: True positives - correctly predicted positive cases.
        tn: True negatives - correctly predicted negative cases.
        fp: False positives - negative cases predicted as positive.
        fn: False negatives - positive cases predicted as negative.
        precision: Of all positive predictions, how many were correct.
        recall: Of all target positives, how many were correctly predicted.
        f1_score: Harmonic mean of precision and recall.
        specificity: Of all target negatives, how many were correctly predicted.
    """

    n_samples: int
    accuracy: float
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1_score: float
    specificity: float

    def to_str(self) -> str:
        """Return a human-friendly text representation of the
        classification results.

        Returns:
            A formatted string with progress bars for each metric and a confusion matrix.
        """
        metrics = [
            ("Accuracy", self.accuracy),
            ("Precision", self.precision),
            ("Recall", self.recall),
            ("Specificity", self.specificity),
            ("F1 Score", self.f1_score),
        ]

        header = f"Classification Results (n={self.n_samples})"
        separator = "-" * len(header)

        metric_lines = [
            f"{name:<11} {make_bar(value, length=20)}  {value:.4f}" for name, value in metrics
        ]
        metric_lines[0] = metric_lines[0] + f"  ({self.tp + self.tn:,}/{self.n_samples:,})"
        metric_lines[1] = metric_lines[1] + f"  ({self.tp:,}/{self.tp + self.fp:,})"
        metric_lines[2] = metric_lines[2] + f"  ({self.tp:,}/{self.tp + self.fn:,})"
        metric_lines[3] = metric_lines[3] + f"  ({self.tn:,}/{self.tn + self.fp:,})"
        metric_text = "\n".join(metric_lines)

        confusion = f"Confusion Matrix: TP={self.tp}  TN={self.tn}  FP={self.fp}  FN={self.fn}"

        return f"{header}\n{separator}\n{metric_text}\n\n{confusion}"


def compute_binary_classification_metrics(
    df: pl.DataFrame,
    target_col: str,
    predict_col: str,
) -> BinaryClassificationResults:
    """Compute accuracy, confusion matrix, and core classification
    metrics from a Polars DataFrame.

    Args:
        df: A Polars DataFrame containing the target and predicted columns.
        target_col: The name of the column containing the target binary values.
        predict_col: The name of the column containing the predicted binary values.

    Returns:
        A ClassificationResults dataclass containing n_samples, accuracy, TP, TN, FP, FN,
        precision, recall, F1 score, and specificity.

    Raises:
        ValueError: If the DataFrame is empty or the columns contain non-binary values.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.metrics import compute_binary_classification_metrics
        >>> df = pl.DataFrame({"target": [1, 0, 1, 0], "predicted": [1, 0, 0, 1]})
        >>> results = compute_binary_classification_metrics(df, target_col="target", predict_col="predicted")
        >>> results.n_samples
        4
        >>> results.accuracy
        0.5
    """
    if df.is_empty():
        msg = "DataFrame is empty."
        raise ValueError(msg)

    for col in [target_col, predict_col]:
        unique_values = df[col].unique().to_list()
        if not set(unique_values).issubset({0, 1}):
            msg = f"Column '{col}' contains non-binary values: {unique_values}"
            raise ValueError(msg)

    target = df[target_col].cast(pl.Int8)
    predicted = df[predict_col].cast(pl.Int8)

    n_samples = len(df)
    tp = int(((target == 1) & (predicted == 1)).sum())
    tn = int(((target == 0) & (predicted == 0)).sum())
    fp = int(((target == 0) & (predicted == 1)).sum())
    fn = int(((target == 1) & (predicted == 0)).sum())

    accuracy = (tp + tn) / n_samples
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return BinaryClassificationResults(
        n_samples=n_samples,
        accuracy=accuracy,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        specificity=specificity,
    )
