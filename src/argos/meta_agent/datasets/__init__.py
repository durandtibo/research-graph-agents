r"""Contain abstractions and implementations for datasets."""

from __future__ import annotations

__all__ = ["BaseDataset", "Dataset"]

from argos.meta_agent.datasets.base import BaseDataset
from argos.meta_agent.datasets.vanilla import Dataset
