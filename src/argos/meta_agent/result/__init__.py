r"""Contain the metrics results."""

from __future__ import annotations

__all__ = ["BaseResult", "Result", "ResultDict"]

from argos.meta_agent.result.base import BaseResult
from argos.meta_agent.result.mapping import ResultDict
from argos.meta_agent.result.vanilla import Result
