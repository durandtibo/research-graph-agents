r"""Contain the implementation of a benchmark example."""

from __future__ import annotations

__all__ = ["Benchmark", "BenchmarkExample", "PredictionRecord", "PredictionResult"]

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

InputT = TypeVar("InputT")
TargetT = TypeVar("TargetT")
PredictionT = TypeVar("PredictionT")


@dataclass
class BenchmarkExample(Generic[InputT, TargetT]):
    r"""Define the benchmark example class."""

    id: str
    input: InputT
    target: TargetT
    metadata: dict[str, Any] | None = None


@dataclass
class Benchmark(Generic[InputT, TargetT]):
    r"""Define the benchmark class.

    The examples are indexed by their IDs.
    """

    examples: dict[str, BenchmarkExample[InputT, TargetT]]
    metadata: dict[str, Any] | None = None


@dataclass
class PredictionRecord(Generic[PredictionT]):
    r"""Define the prediction record class."""

    example_id: str
    prediction: PredictionT
    metadata: dict[str, Any] | None = None


@dataclass
class PredictionResult(Generic[PredictionT]):
    r"""Define the prediction result class."""

    records: list[PredictionRecord[PredictionT]]
