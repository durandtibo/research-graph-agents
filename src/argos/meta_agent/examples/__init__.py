r"""Contain abstractions and implementations for examples.

An example is a single labeled instance in a dataset, pairing an
``input`` with a ground-truth ``target``. This sub-package also
provides utilities for converting between lists of examples and Polars
DataFrames.
"""

from __future__ import annotations

__all__ = ["BaseExample", "Example", "dataframe_to_examples", "examples_to_dataframe"]

from argos.meta_agent.examples.base import BaseExample
from argos.meta_agent.examples.utils import dataframe_to_examples, examples_to_dataframe
from argos.meta_agent.examples.vanilla import Example
