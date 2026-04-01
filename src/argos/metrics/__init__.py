r"""Contain metrics to evaluate agent outputs.

This sub-package provides functions and result containers for
computing evaluation metrics on tasks such as classification.

"""

from __future__ import annotations

__all__ = ["BinaryClassificationResults", "compute_binary_classification_metrics"]

from argos.metrics.binary_classification import (
    BinaryClassificationResults,
    compute_binary_classification_metrics,
)
