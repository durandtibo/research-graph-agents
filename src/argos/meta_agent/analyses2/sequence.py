r"""Implement a simple analysis implementation."""

from __future__ import annotations

__all__ = ["AnalysisList"]

from dataclasses import dataclass

from coola.equality import objects_are_equal
from coola.utils.format import (
    repr_indent,
    repr_sequence,
    str_indent,
    str_sequence,
)

from argos.meta_agent.analyses2.base import BaseAnalysis, PrimitiveType


@dataclass(frozen=True)
class AnalysisList(BaseAnalysis):
    r"""Implement an analysis that combines a list of analyses.

    Args:
        analyses: The list of analysis objects to combine.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses2 import Analysis, AnalysisList
        >>> analysis = AnalysisList([Analysis("style analysis"), Analysis("semantic analysis")])
        >>> analysis
        AnalysisList(
          (0): Analysis(content='style analysis', metadata=None)
          (1): Analysis(content='semantic analysis', metadata=None)
        )
        >>> analysis.to_primitive()
        ['style analysis', 'semantic analysis']
        >>> print(analysis.to_json(indent=2))
        [
          "style analysis",
          "semantic analysis"
        ]

        ```
    """

    analyses: list[BaseAnalysis]

    def __repr__(self) -> str:
        if not self.analyses:
            return f"{self.__class__.__qualname__}()"

        args = repr_indent(repr_sequence(self.analyses))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        if not self.analyses:
            return f"{self.__class__.__qualname__}()"

        args = str_indent(str_sequence(self.analyses))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.analyses, other.analyses, equal_nan=equal_nan)

    def to_primitive(self) -> PrimitiveType:
        return [value.to_primitive() for value in self.analyses]
