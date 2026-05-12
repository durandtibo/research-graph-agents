r"""Data containers exchanged by the meta-agent optimization loop.

This package provides immutable, typed entities that capture model
inputs, expected outputs, predictions, and combined records. They are
designed for easy serialization via ``to_dict``/``from_dict`` and for
stable equality checks during evaluations.

Example:
    ```pycon
    >>> from argos.meta_agent.entities import Record
    >>> record = Record(id="q1", input="What is 2+2?", target="4", prediction="5")
    >>> record.to_dict()
    {'id': 'q1', 'input': 'What is 2+2?', 'target': '4', 'prediction': '5', 'metadata': None}

    ```
"""

from __future__ import annotations

__all__ = [
    "BaseEntity",
    "BaseExample",
    "BaseLabeledExample",
    "BasePrediction",
    "BaseRecord",
    "Example",
    "LabeledExample",
    "Prediction",
    "Record",
]

from argos.meta_agent.entities.base import BaseEntity
from argos.meta_agent.entities.example import BaseExample, Example
from argos.meta_agent.entities.labeled_example import BaseLabeledExample, LabeledExample
from argos.meta_agent.entities.prediction import BasePrediction, Prediction
from argos.meta_agent.entities.record import BaseRecord, Record
