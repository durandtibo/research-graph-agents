r"""Contain abstractions and implementations for evaluators.

This sub-package provides the abstract base class for evaluators
and concrete implementations that compare agent predictions
against benchmark targets to compute evaluation metrics.
"""

from __future__ import annotations

__all__ = ["BaseEvaluator", "NoOpEvaluator"]

from argos.meta_agent.evaluators.base import BaseEvaluator
from argos.meta_agent.evaluators.noop import NoOpEvaluator
