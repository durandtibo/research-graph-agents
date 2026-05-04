r"""Contain abstractions and implementations for metric results.

This sub-package provides the abstract base class for result containers
and concrete implementations for storing, formatting, and comparing
evaluation metrics in various representations.
"""

from __future__ import annotations

__all__ = ["BaseResult", "Result", "ResultDict"]

from argos.meta_agent.results.base import BaseResult
from argos.meta_agent.results.mapping import ResultDict
from argos.meta_agent.results.vanilla import Result
