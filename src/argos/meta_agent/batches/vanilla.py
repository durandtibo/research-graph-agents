r"""Contain a simple implementation of a batch."""

from __future__ import annotations

__all__ = ["Batch"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, TypeVar

from coola.equality import objects_are_equal

from argos.meta_agent.batches.base import BaseBatch
from argos.meta_agent.entities import BaseEntity

if TYPE_CHECKING:
    from collections.abc import Sequence


T = TypeVar("T", bound=BaseEntity)


@dataclass(frozen=True)
class Batch(BaseBatch[T]):
    r"""Implement a concrete batch of items.

    The items are indexed by their IDs.

    Attributes:
        items: A mapping from example ID to
            :class:`BaseEntity` instance.
        metadata: Optional dictionary of auxiliary information about
            the dataset. Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.batches import Batch
        >>> from argos.meta_agent.entities import LabeledExample
        >>> dataset = Batch(
        ...     items={
        ...         "q1": LabeledExample(id="q1", input="What is 2+2?", target="4"),
        ...         "q2": LabeledExample(id="q2", input="What is 3+3?", target="6"),
        ...     }
        ... )
        >>> len(dataset.items)
        2

        ```
    """

    items: dict[str, T]
    metadata: dict[str, Any] | None = None

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(
            self.items, other.items, equal_nan=equal_nan
        ) and objects_are_equal(self.metadata, other.metadata, equal_nan=equal_nan)

    @classmethod
    def from_list(
        cls,
        items: Sequence[T],
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        data = {example.id: example for example in items}
        if len(data) != len(items):
            msg = "Some example IDs are duplicated"
            raise ValueError(msg)
        return cls(items=data, metadata=metadata)
