r"""Contain datasets used for benchmarking and evaluation.

This sub-package provides curated datasets with labeled examples for
training, evaluating, and benchmarking LLM-powered agents.

"""

from __future__ import annotations

__all__ = ["generate_haiku_dataset"]

from argos.datasets.haiku import generate_haiku_dataset
