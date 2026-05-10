r"""Contain a simple implementation of a prediction."""

from __future__ import annotations

__all__ = ["Prediction"]

from dataclasses import dataclass
from typing import Any, Self, TypeVar

from coola.equality import objects_are_equal

from argos.meta_agent.predictions.base import BasePrediction

T = TypeVar("T")


@dataclass(frozen=True)
class Prediction(BasePrediction[T]):
    r"""Store a single prediction paired with its example identifier.

    Args:
        example_id: The identifier of the benchmark example this
            prediction corresponds to.
        prediction: The prediction produced by the agent.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.predictions import Prediction
        >>> record = Prediction(example_id="q1", prediction="4")
        >>> record.example_id
        'q1'
        >>> record.prediction
        '4'

        ```
    """

    example_id: str
    prediction: T
    metadata: dict[str, Any] | None = None

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.to_dict(), other.to_dict(), equal_nan=equal_nan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "prediction": self.prediction,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            example_id=data["example_id"],
            prediction=data["prediction"],
            metadata=data.get("metadata"),
        )
