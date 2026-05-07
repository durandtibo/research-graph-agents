r"""Contain abstractions and implementations for analyses.

An analysis encapsulates the diagnostic information produced by an
analyzer, exposing it as text, a plain dictionary, or other serializable
formats.
"""

from __future__ import annotations

__all__ = ["Analysis", "AnalysisDict", "BaseAnalysis", "IndentedListAnalysisDict"]

from argos.meta_agent.analyses.base import BaseAnalysis
from argos.meta_agent.analyses.mapping import AnalysisDict, IndentedListAnalysisDict
from argos.meta_agent.analyses.vanilla import Analysis
