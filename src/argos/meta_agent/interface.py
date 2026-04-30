r"""Contain the implementation of a benchmark example."""

from __future__ import annotations

__all__ = ["Benchmark", "BenchmarkExample", "PredictionRecord", "PredictionResult"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    @classmethod
    def from_predictions(
        cls, example_ids: Sequence[str], predictions: Sequence[PredictionT]
    ) -> PredictionResult[PredictionT]:
        r"""Create a prediction result from a list of example IDs and
        predictions.

        example_ids and predictions must be aligned i.e. the n-th
        example ID should match the n-th prediction.

        Args:
            example_ids: A list of example IDs
            predictions: A list of predictions

        Returns:
            The prediction result.
        """
        if len(example_ids) != len(predictions):
            msg = "example_ids and predictions must have the same length"
            raise ValueError(msg)
        return PredictionResult(
            [
                PredictionRecord(example_id=example_id, prediction=prediction)
                for example_id, prediction in zip(example_ids, predictions)
            ]
        )

    @classmethod
    def from_dict(cls, data: dict[str, PredictionT]) -> PredictionResult[PredictionT]:
        r"""Create a prediction result from a dictionary.

        Args:
            data: A dictionary containing example IDs and predictions.

        Returns:
            The prediction result.
        """
        return cls.from_predictions(example_ids=data.keys(), predictions=data.values())

    def to_dict(self) -> dict[str, PredictionT]:
        r"""Return a dictionary containing example IDs and predictions.

        Returns:
            A dictionary containing example IDs and predictions.
        """
        return {record.example_id: record.prediction for record in self.records}
