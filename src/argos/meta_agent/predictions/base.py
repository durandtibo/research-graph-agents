r"""Contain the base class to define a prediction."""

from __future__ import annotations

__all__ = ["BasePrediction"]

from abc import ABC, abstractmethod
from typing import Any, Generic, Self, TypeVar

from coola.equality.tester import EqualNanEqualityTester, get_default_registry

T = TypeVar("T")


class BasePrediction(ABC, Generic[T]):
    r"""Abstract base class defining the interface for a single
    prediction.

    Subclasses must define all attributes and implement all methods.

    Attributes:
        example_id: The identifier of the example this
            prediction corresponds to.
        prediction: The prediction produced by the agent.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.
    """

    example_id: str
    prediction: T
    metadata: dict[str, Any] | None = None

    @abstractmethod
    def equal(self, other: object, equal_nan: bool = False) -> bool:
        r"""Return ``True`` if the two objects are equal, otherwise
        ``False``.

        Args:
            other: The value to compare with.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        r"""Serialise the example to a plain dictionary.

        Returns:
            A dictionary with keys ``id``, ``input``, ``target``, and
            ``metadata``.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        r"""Construct an instance from a plain dictionary.

        Args:
            data: Must contain ``id``, ``input``, and ``target`` keys.
                ``metadata`` is optional.

        Returns:
            A new instance of the calling subclass.
        """


get_default_registry().register_many({BasePrediction: EqualNanEqualityTester()}, exist_ok=True)
