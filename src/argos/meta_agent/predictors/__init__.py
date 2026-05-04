r"""Contain abstractions and implementations for predictors.

This sub-package provides the abstract base class for predictors and
concrete implementations that run an agent over all examples in a
benchmark and collect the prediction results.
"""

from __future__ import annotations

__all__ = ["BasePredictor", "BatchPredictor"]

from argos.meta_agent.predictors.base import BasePredictor
from argos.meta_agent.predictors.batch import BatchPredictor
