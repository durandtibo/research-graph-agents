r"""Implement an analyzer that converts a DataFrame to a markdown format
analysis."""

from __future__ import annotations

__all__ = ["Data2MarkdownAnalyzer"]

from typing import TYPE_CHECKING

from argos.meta_agent.analyses import Analysis
from argos.meta_agent.analyzers.base import BaseAnalyzer
from argos.utils.dataframe import dataframe_to_markdown

if TYPE_CHECKING:
    import polars as pl


class Data2MarkdownAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer that converts a DataFrame to a markdown
    format."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def analyze(self, data: pl.DataFrame) -> Analysis:
        return Analysis(dataframe_to_markdown(data))

    def equal(self, other: object, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return type(other) is type(self)
