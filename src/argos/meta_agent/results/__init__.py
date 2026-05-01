r"""Contain the metrics results."""

from __future__ import annotations

__all__ = ["BaseResult", "Result", "ResultDict"]

from argos.meta_agent.results.base import BaseResult
from argos.meta_agent.results.mapping import ResultDict
from argos.meta_agent.results.vanilla import Result
