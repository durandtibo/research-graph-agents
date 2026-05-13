r"""Contain abstractions and implementations for batches.

A dataset is an indexed collection of labeled examples used to run and
evaluate an agent.
"""

from __future__ import annotations

__all__ = ["BaseBatch", "Batch"]

from argos.meta_agent.batches.base import BaseBatch
from argos.meta_agent.batches.vanilla import Batch
