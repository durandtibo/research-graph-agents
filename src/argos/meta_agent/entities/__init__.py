r"""Contain entities."""

from __future__ import annotations

__all__ = [
    "BaseEntity",
    "BaseExample",
    "BaseLabeledExample",
    "BaseRecord",
    "Example",
    "LabeledExample",
    "Record",
]

from argos.meta_agent.entities.base import BaseEntity
from argos.meta_agent.entities.example import BaseExample, Example
from argos.meta_agent.entities.labeled_example import BaseLabeledExample, LabeledExample
from argos.meta_agent.entities.record import BaseRecord, Record
