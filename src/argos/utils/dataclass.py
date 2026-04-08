r"""Contain utility functions for dataclasses."""

from __future__ import annotations

__all__ = ["dataclass_to_dict"]

from dataclasses import asdict, is_dataclass
from typing import Any

from coola.recursive import recursive_apply


def dataclass_to_dict(obj: Any) -> Any:
    r"""Convert each dataclass object to its associated dictionary
    representation using ``dataclass.asdict``.

    Args:
        obj: The object to be converted, can be nested.

    Returns:
        The converted object where dataclasses are replaced by dictionaries.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from argos.utils.dataclass import dataclass_to_dict
        >>> @dataclass
        ... class Point:
        ...     x: int
        ...     y: int
        ...
        >>> dataclass_to_dict(Point(x=1, y=2))
        {'x': 1, 'y': 2}
        >>> dataclass_to_dict([Point(x=1, y=2), Point(x=3, y=4)])
        [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]

        ```
    """
    return recursive_apply(obj, lambda x: asdict(x) if is_dataclass(x) else x)
