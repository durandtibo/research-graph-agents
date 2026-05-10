r"""Implement a simple analysis implementation."""

from __future__ import annotations

__all__ = ["Analysis"]

from dataclasses import dataclass
from typing import Any, Self

from coola.equality import objects_are_equal

from argos.meta_agent.analyses.base import BaseAnalysis


@dataclass(frozen=True)
class Analysis(BaseAnalysis):
    r"""Define a simple analysis.

    Args:
        content: The human-readable analysis content. The value is
            returned unchanged by :meth:`to_text`.
        metadata: Optional auxiliary information associated with the
            analysis.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses import Analysis
        >>> analysis = Analysis("my custom analysis: blabla...")
        >>> analysis.to_dict()
        {'content': 'my custom analysis: blabla...', 'metadata': None}
        >>> analysis.to_text()
        'my custom analysis: blabla...'

        ```
    """

    content: str
    metadata: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(content_len={len(self.content)}, metadata={self.metadata})"

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.to_dict(), other.to_dict(), equal_nan=equal_nan)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            content=data["content"],
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "metadata": self.metadata}

    def to_text(self) -> str:
        if not self.content:
            return "_Empty analysis_"
        return self.content
