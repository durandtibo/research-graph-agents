r"""Implement a simple analysis implementation."""

from __future__ import annotations

__all__ = ["Analysis"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

from coola.equality import objects_are_equal

from argos.meta_agent.analyses.base import BaseAnalysis

if TYPE_CHECKING:
    from argos.meta_agent.typing import FlatDict


@dataclass(frozen=True)
class Analysis(BaseAnalysis):
    r"""Define a simple analysis.

    Args:
        content: The content of the analysis in a markdown format.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses import Analysis
        >>> analysis = Analysis("my custom analysis: blabla...")
        >>> analysis.to_dict()
        {'content': 'my custom analysis: blabla...'}
        >>> analysis.to_markdown()
        'my custom analysis: blabla...'

        ```
    """

    content: str

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(content_len={len(self.content)})"

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.content, other.content, equal_nan=equal_nan)

    def to_dict(self) -> FlatDict:
        return {"content": self.content}

    def to_markdown(self) -> str:
        if not self.content:
            return "_Empty analysis_"
        return self.content
