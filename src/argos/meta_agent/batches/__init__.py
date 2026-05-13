r"""Contain abstractions and implementations for batches.

A batch is an indexed collection of entities keyed by their IDs.
Batches are the primary data structure used to pass groups of examples
or records through the meta-agent pipeline.
"""

from __future__ import annotations

__all__ = ["BaseBatch", "Batch"]

from argos.meta_agent.batches.base import BaseBatch
from argos.meta_agent.batches.vanilla import Batch
