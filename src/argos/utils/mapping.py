r"""Contain utility functions for mappings."""

from __future__ import annotations

__all__ = ["to_dict"]

from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel


def to_dict(data: Any) -> dict[Any, Any]:
    r"""Convert an object to a dictionary when possible.

    Args:
        data: The object to convert to a dictionary.

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
    if isinstance(data, BaseModel):
        return data.model_dump()
    if is_dataclass(data):
        return asdict(data)
    return data
