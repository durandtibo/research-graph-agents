r"""Define the abstract class and implementations for records."""

from __future__ import annotations

__all__ = ["BasePrediction", "Prediction"]

from dataclasses import asdict, dataclass
from typing import Any, Generic, Self, TypeVar

from argos.meta_agent.entities.base import BaseEntity

T = TypeVar("T")


class BasePrediction(BaseEntity, Generic[T]):
    r"""Abstract base class defining the interface for a single labeled
    example.

    Subclasses must define all attributes and implement all methods.

    Attributes:
        id: A unique identifier for the example.
        prediction: The predicted output.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.entities import Prediction
        >>> example = Prediction(id="q1", prediction="5")
        >>> example
        Prediction(id='q1', prediction='5', metadata=None)
        >>> example.id
        'q1'
        >>> example.prediction
        '5'

        ```
    """

    prediction: T


@dataclass(frozen=True)
class Prediction(BasePrediction[T]):
    r"""Define a concrete labeled example for use in datasets.

    Args:
        id: A unique identifier for the example.
        prediction: The predicted output.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.entities import Prediction
        >>> example = Prediction(id="q1", prediction="5")
        >>> example
        Prediction(id='q1', prediction='5', metadata=None)
        >>> example.id
        'q1'
        >>> example.prediction
        '5'

        ```
    """

    id: str
    prediction: T
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            prediction=data["prediction"],
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
