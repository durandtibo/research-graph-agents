r"""Contain abstractions and implementations for analyses.

An analysis encapsulates the diagnostic information produced by an
analyzer, exposing it as text, a plain dictionary, or other serializable
formats.
"""

from __future__ import annotations

__all__ = ["Analysis", "AnalysisDict", "BaseAnalysis"]

from argos.meta_agent.analyses2.base import BaseAnalysis
from argos.meta_agent.analyses2.mapping import AnalysisDict
from argos.meta_agent.analyses2.vanilla import Analysis
