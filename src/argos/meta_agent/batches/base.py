r"""Contain the base class to define a dataset."""

from __future__ import annotations

__all__ = ["BaseBatch"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

import polars as pl
from coola.equality.tester import EqualNanEqualityTester, get_default_registry

from argos.meta_agent.entities import BaseEntity, Record, dataframe_to_entities

if TYPE_CHECKING:
    from collections.abc import Sequence


T = TypeVar("T", bound=BaseEntity)


class BaseBatch(ABC, Generic[T]):
    r"""Abstract base class defining the interface for a dataset.

    Subclasses must define all attributes and implement all methods.

    Attributes:
        items: A mapping from entity ID to
            :class:`~argos.meta_agent.entities.BaseEntity` instance.
        metadata: Optional dictionary of auxiliary information about
            the dataset. Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.batches import Batch
        >>> from argos.meta_agent.entities import LabeledExample
        >>> dataset = Batch(
        ...     {
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

    def to_dataframe(self) -> pl.DataFrame:
        r"""Return a Polars DataFrame representing the batch of items.

        The dataset-level ``metadata`` attribute is not included in the
        returned DataFrame. Each row corresponds to one example, with
        columns derived from the example's ``to_dict`` representation.

        Returns:
            A Polars DataFrame with one row per example.

        Example:
            ```pycon
            >>> import polars as pl
            >>> from argos.meta_agent.batches import Batch
            >>> from argos.meta_agent.entities import LabeledExample
            >>> dataset = Batch.from_list(
            ...     [
            ...         LabeledExample(id="q1", input="What is 2+2?", target="4"),
            ...         LabeledExample(id="q2", input="What is 3+3?", target="6"),
            ...     ]
            ... )
            >>> dataset.to_dataframe()
            shape: (2, 4)
            ┌─────┬──────────────┬────────┬──────────┐
            │ id  ┆ input        ┆ target ┆ metadata │
            │ --- ┆ ---          ┆ ---    ┆ ---      │
            │ str ┆ str          ┆ str    ┆ null     │
            ╞═════╪══════════════╪════════╪══════════╡
            │ q1  ┆ What is 2+2? ┆ 4      ┆ null     │
            │ q2  ┆ What is 3+3? ┆ 6      ┆ null     │
            └─────┴──────────────┴────────┴──────────┘

            ```
        """
        return pl.DataFrame([r.to_dict() for r in self.items.values()])

    @classmethod
    def from_dataframe(
        cls,
        frame: pl.DataFrame,
        metadata: dict[str, Any],
        entity_type: type[T] = Record,
    ) -> Self:
        r"""Create a dataset from a dataframe.

        Args:
            frame: A Polars DataFrame where each row represents a single
                example. Column names must match the fields expected by
                ``example_type.from_dict``.
            metadata: The dataset metadata.
            entity_type: The example class to instantiate for each row.
                Defaults to :class:`Example`.

        Returns:
            The dataset instance.
        """
        items = dataframe_to_entities(frame=frame, entity_type=entity_type)
        return cls.from_list(items=items, metadata=metadata)

    @classmethod
    @abstractmethod
    def from_list(
        cls,
        items: Sequence[T],
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        r"""Create a dataset from a list of items.

        Args:
            items: A list of items. The example IDs must be unique.
            metadata: The dataset metadata.

        Returns:
            The dataset instance.

        Raises:
            ValueError: If any example IDs are duplicated.

        Example:
            ```pycon
            >>> from argos.meta_agent.batches import Batch
            >>> from argos.meta_agent.entities import LabeledExample
            >>> dataset = Batch.from_list(
            ...     [
            ...         LabeledExample(id="q1", input="What is 2+2?", target="4"),
            ...         LabeledExample(id="q2", input="What is 3+3?", target="6"),
            ...     ]
            ... )
            >>> len(dataset.items)
            2

            ```
        """


get_default_registry().register_many({BaseBatch: EqualNanEqualityTester()}, exist_ok=True)
