r"""Contain abstractions and implementations for analyzers.

An analyzer inspects a Polars DataFrame of prediction results and
returns a :class:`~argos.meta_agent.analyses.BaseAnalysis` object.
Concrete implementations include no-op, data-to-text converters, an
LLM-based refinement wrapper, and a JSON export decorator.
"""

from __future__ import annotations

__all__ = [
    "Analyzer",
    "BaseAnalyzer",
    "BaseExportAnalyzer",
    "Data2CsvAnalyzer",
    "Data2MarkdownAnalyzer",
    "Data2StrAnalyzer",
    "JsonExportAnalyzer",
    "NoOpAnalyzer",
    "RefinedAnalyzer",
]

from argos.meta_agent.analyzers.base import BaseAnalyzer
from argos.meta_agent.analyzers.data2csv import Data2CsvAnalyzer
from argos.meta_agent.analyzers.data2markdown import Data2MarkdownAnalyzer
from argos.meta_agent.analyzers.data2str import Data2StrAnalyzer
from argos.meta_agent.analyzers.export import JsonExportAnalyzer
from argos.meta_agent.analyzers.noop import NoOpAnalyzer
from argos.meta_agent.analyzers.refined import RefinedAnalyzer
from argos.meta_agent.analyzers.vanilla import Analyzer
