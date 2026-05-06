r"""Define a analyzer wrapper to export the analysis to a json file."""

from __future__ import annotations

__all__ = ["JsonExportAnalyzer"]

import logging
from typing import TYPE_CHECKING

from iden.io import save_json

from argos.meta_agent.analyzers.base import BaseAnalyzer

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from argos.meta_agent.analyses import BaseAnalysis

logger: logging.Logger = logging.getLogger(__name__)


class JsonExportAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer wrapper that persists the analysis to a
    JSON file.

    Delegates the analysis to an inner ``analyzer``, then serialises
    the result via :meth:`~argos.meta_agent.analyses.BaseAnalysis.to_dict`
    and saves it to ``path``.  The original analysis object is returned
    unchanged so the wrapper is transparent to callers.

    Args:
        analyzer: The inner analyzer whose output is saved.
        path: Destination path for the JSON file.

    Example:
        ```pycon
        >>> import pathlib, tempfile, polars as pl
        >>> from argos.meta_agent.analyses import Analysis
        >>> from argos.meta_agent.analyzers import Analyzer, JsonExportAnalyzer
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = pathlib.Path(tmp) / "analysis.json"
        ...     inner = Analyzer(Analysis("my analysis"))
        ...     analyzer = JsonExportAnalyzer(analyzer=inner, path=path)
        ...     analysis = analyzer.analyze(pl.DataFrame())
        ...     analysis.to_text()
        ...
        'my analysis'

        ```
    """

    def __init__(self, analyzer: BaseAnalyzer, path: Path) -> None:
        self._analyzer = analyzer
        self._path = path

    def analyze(self, data: pl.DataFrame) -> BaseAnalysis:
        logger.info("Analyzing the data...")
        analysis = self._analyzer.analyze(data)
        logger.info(f"Exporting the analysis to a json file: {self._path}")
        save_json(analysis.to_dict(), self._path)
        return analysis

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return (
            self._analyzer.equal(other._analyzer, equal_nan=equal_nan) and self._path == other._path
        )
