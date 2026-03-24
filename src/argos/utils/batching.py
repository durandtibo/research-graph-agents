r"""Contain utility functions for batching."""

from __future__ import annotations

__all__ = ["batchify"]

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


def batchify(items: Sequence[T], *, size: int) -> list[tuple[T, ...]]:
    r"""Generate batches of items.

    Args:
        items: List of items to batch.
        size: Size of each batch.

    Returns:
        Batched list of items.
    """
    if size < 1:
        msg = "size must be >= 1"
        raise ValueError(msg)
    return [tuple(items[i : i + size]) for i in range(0, len(items), size)]
