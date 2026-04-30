r"""Contain the implementation of a benchmark example."""

from __future__ import annotations

__all__ = ["PredictionRecord", "PredictionResult"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence


T = TypeVar("T")


@dataclass
class PredictionRecord(Generic[T]):
    r"""Define the prediction record class."""

    example_id: str
    prediction: T
    metadata: dict[str, Any] | None = None


@dataclass
class PredictionResult(Generic[T]):
    r"""Define the prediction result class."""

    records: list[PredictionRecord[T]]

    @classmethod
    def from_predictions(
        cls, example_ids: Sequence[str], predictions: Sequence[T]
    ) -> PredictionResult[T]:
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
    def from_dict(cls, data: dict[str, T]) -> PredictionResult[T]:
        r"""Create a prediction result from a dictionary.

        Args:
            data: A dictionary containing example IDs and predictions.

        Returns:
            The prediction result.
        """
        return cls.from_predictions(example_ids=data.keys(), predictions=data.values())

    def to_dict(self) -> dict[str, T]:
        r"""Return a dictionary containing example IDs and predictions.

        Returns:
            A dictionary containing example IDs and predictions.
        """
        return {record.example_id: record.prediction for record in self.records}
