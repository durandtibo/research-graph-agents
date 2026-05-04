r"""Contain the base class to define a dataset example."""

from __future__ import annotations

__all__ = ["BaseExample"]

from abc import ABC, abstractmethod
from typing import Any, Generic, Self

from argos.meta_agent.typing import InputT, TargetT


class BaseExample(ABC, Generic[InputT, TargetT]):
    r"""Abstract base class defining the interface for a single labeled
    example.

    Subclasses must define all attributes and implement all methods.
    """

    id: str
    input: InputT
    target: TargetT
    metadata: dict[str, Any] | None = None

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise the example to a plain dictionary.

        Returns:
            A dictionary with keys ``id``, ``input``, ``target``, and
            ``metadata``.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct an instance from a plain dictionary.

        Args:
            data: Must contain ``id``, ``input``, and ``target`` keys.
                ``metadata`` is optional.

        Returns:
            A new instance of the calling subclass.
        """
