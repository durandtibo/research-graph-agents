r"""Contain abstractions and implementations for predictions."""

from __future__ import annotations

__all__ = ["BasePrediction", "Prediction"]

from argos.meta_agent.predictions.base import BasePrediction
from argos.meta_agent.predictions.vanilla import Prediction
