r"""Contain metrics to evaluate agent outputs.

This sub-package provides functions and result containers for
computing evaluation metrics on tasks such as classification.

Available components:

- :class:`~argos.metrics.binary_classification.BinaryClassificationResults`:
  A dataclass that stores binary classification metrics including
  accuracy, precision, recall, F1 score, and specificity.
- :func:`~argos.metrics.binary_classification.compute_binary_classification_metrics`:
  Compute binary classification metrics from a Polars DataFrame with
  target and predicted columns.
"""

from __future__ import annotations

__all__ = ["BinaryClassificationResults", "compute_binary_classification_metrics"]

from argos.metrics.binary_classification import (
    BinaryClassificationResults,
    compute_binary_classification_metrics,
)
