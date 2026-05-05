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
    r"""Define a analyzer wrapper to export the analysis to a json file.

    Args:
        analyzer: An analyzer to save the analysis to a json file.
        path: Path to save the analysis to a json file.
    """

    def __init__(self, analyzer: BaseAnalyzer, path: Path) -> None:
        self._analyzer = analyzer
        self._path = path

    def analyze(self, data: pl.DataFrame) -> BaseAnalysis:
        analysis = self._analyzer.analyze(data)
        logger.info(f"Saving the analysis to json: {self._path}")
        save_json(analysis.to_dict(), self._path)
        return analysis

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return (
            self._analyzer.equal(other._analyzer, equal_nan=equal_nan) and self._path == other._path
        )
