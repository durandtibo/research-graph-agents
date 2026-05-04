r"""Contain the base class to define a dataset example."""

from __future__ import annotations

__all__ = ["BaseExample"]

from abc import ABC
from typing import Any, Generic

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

    def to_dict(self) -> dict[str, Any]:
        """Serialise the example to a plain dictionary.

        Returns:
            A dictionary with keys ``id``, ``input``, ``target``, and
            ``metadata``.
        """
        return {
            "id": self.id,
            "input": self.input,
            "target": self.target,
            "metadata": self.metadata,
        }
