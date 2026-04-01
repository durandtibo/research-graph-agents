r"""Contain datasets used for benchmarking and evaluation.

This sub-package provides curated datasets with labeled examples for
training, evaluating, and benchmarking LLM-powered agents.

Available datasets:

- :func:`~argos.datasets.haiku.generate_haiku_dataset`: A labeled
  dataset of haiku examples with both positive (valid) and negative
  (invalid) samples across multiple topics. Each row includes the
  topic, the haiku text, individual judgement targets for structure
  and topic adherence, and a final binary target label.
"""

from __future__ import annotations

__all__ = ["generate_haiku_dataset"]

from argos.datasets.haiku import generate_haiku_dataset
