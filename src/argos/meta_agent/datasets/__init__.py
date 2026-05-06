r"""Contain abstractions and implementations for datasets.

A dataset is an indexed collection of labeled examples used to run
and evaluate an agent. Concrete implementations back the collection
with a flat dictionary of :class:`~argos.meta_agent.examples.BaseExample`
instances keyed by their IDs.
"""

from __future__ import annotations

__all__ = ["BaseDataset", "Dataset"]

from argos.meta_agent.datasets.base import BaseDataset
from argos.meta_agent.datasets.vanilla import Dataset
