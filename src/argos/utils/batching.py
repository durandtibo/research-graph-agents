r"""Contain utility functions for batching."""

from __future__ import annotations

__all__ = ["batchify"]

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


def batchify(items: Sequence[T], *, size: int) -> list[tuple[T, ...]]:
    r"""Generate batches of items from a sequence.

    Splits ``items`` into consecutive non-overlapping tuples of length
    ``size``. The last batch may be shorter than ``size`` if the
    sequence length is not evenly divisible.

    Args:
        items: The sequence of items to batch.
        size: The maximum number of items per batch. Must be >= 1.

    Returns:
        A list of tuples, each containing at most ``size`` items. An
            empty list is returned when ``items`` is empty.

    Raises:
        ValueError: If ``size`` is less than 1.

    Example:
        ```pycon
        >>> from argos.utils.batching import batchify
        >>> batchify([1, 2, 3, 4, 5], size=2)
        [(1, 2), (3, 4), (5,)]
        >>> batchify([], size=3)
        []

        ```
    """
    if size < 1:
        msg = "size must be >= 1"
        raise ValueError(msg)
    return [tuple(items[i : i + size]) for i in range(0, len(items), size)]
