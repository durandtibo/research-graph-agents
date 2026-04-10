r"""Contain utility functions for mappings."""

from __future__ import annotations

__all__ = ["recursive_to_dict", "to_dict"]

from dataclasses import asdict, is_dataclass
from typing import Any

from coola.recursive import recursive_apply
from pydantic import BaseModel


def to_dict(obj: Any) -> Any:
    r"""Convert an object to a dictionary when possible.

    Args:
        obj: The object to convert to a dictionary.

    Returns:
        The converted dictionary if possible, otherwise the original object.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from argos.utils.mapping import to_dict
        >>> @dataclass
        ... class Point:
        ...     x: int
        ...     y: int
        ...
        >>> to_dict(Point(x=1, y=2))
        {'x': 1, 'y': 2}
        >>> to_dict({"x": 1, "y": 2})
        {'x': 1, 'y': 2}

        ```
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def recursive_to_dict(obj: Any) -> Any:
    r"""Convert each object to its associated dictionary representation
    when possible.

    Args:
        obj: The object to be converted, can be nested.

    Returns:
        The converted object.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from argos.utils.mapping import recursive_to_dict
        >>> @dataclass
        ... class Point:
        ...     x: int
        ...     y: int
        ...
        >>> recursive_to_dict(Point(x=1, y=2))
        {'x': 1, 'y': 2}
        >>> recursive_to_dict([Point(x=1, y=2), Point(x=3, y=4)])
        [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]

        ```
    """
    return recursive_apply(obj, to_dict)
