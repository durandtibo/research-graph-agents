r"""Contain abstractions and implementations for predictions.

A prediction pairs a single agent output with the identifier of the
benchmark example it corresponds to. This sub-package provides the
abstract base class and a concrete frozen-dataclass implementation.
"""

from __future__ import annotations

__all__ = ["BasePrediction", "Prediction"]

from argos.meta_agent.predictions.base import BasePrediction
from argos.meta_agent.predictions.vanilla import Prediction
