r"""Contain metrics to evaluate some tasks."""

from __future__ import annotations

__all__ = ["BinaryClassificationResults", "compute_binary_classification_metrics"]

from argos.metrics.binary_classification import (
    BinaryClassificationResults,
    compute_binary_classification_metrics,
)
