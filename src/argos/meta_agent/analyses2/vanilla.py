r"""Implement a simple analysis implementation."""

from __future__ import annotations

__all__ = ["Analysis"]

from dataclasses import asdict, dataclass
from typing import Any

from coola.equality import objects_are_equal
from coola.utils.format import repr_mapping_line, str_indent, str_mapping

from argos.meta_agent.analyses2.base import BaseAnalysis, PrimitiveType


@dataclass(frozen=True)
class Analysis(BaseAnalysis):
    r"""Define a simple analysis.

    Args:
        content: The content of the analysis.
        metadata: The metadata of the analysis.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses2 import Analysis
        >>> analysis = Analysis("my custom analysis: blabla...")
        >>> analysis
        {'content': 'my custom analysis: blabla...', 'metadata': None}
        >>> analysis.to_primitive()
        'my custom analysis: blabla...'

        ```
    """

    content: PrimitiveType
    metadata: dict[str, Any] | None = None

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {"content": truncate_str(self.content, max_len=50), "metadata": self.metadata}
        )
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = str_indent(str_mapping(asdict(self)))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(asdict(self), asdict(other), equal_nan=equal_nan)

    def to_primitive(self) -> PrimitiveType:
        return self.content
