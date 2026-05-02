r"""Contain the implementation of prediction records and results."""

from __future__ import annotations

__all__ = ["PredictionRecord", "PredictionResult"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence


T = TypeVar("T")


@dataclass
class PredictionRecord(Generic[T]):
    r"""Store a single prediction paired with its example identifier.

    Attributes:
        example_id: The identifier of the benchmark example this
            prediction corresponds to.
        prediction: The prediction produced by the agent.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.
    """

    example_id: str
    prediction: T
    metadata: dict[str, Any] | None = None


@dataclass
class PredictionResult(Generic[T]):
    r"""Store the full set of predictions for a benchmark.

    Attributes:
        records: A list of :class:`PredictionRecord` instances, one
            per benchmark example.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.
    """

    records: list[PredictionRecord[T]]
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_predictions(
        cls,
        example_ids: Sequence[str],
        predictions: Sequence[T],
        metadata: dict[str, Any] | None = None,
    ) -> PredictionResult[T]:
        r"""Create a prediction result from a list of example IDs and predictions.

        ``example_ids`` and ``predictions`` must be aligned, i.e. the
        n-th example ID must match the n-th prediction.

        Args:
            example_ids: A list of example IDs.
            predictions: A list of predictions aligned with
                ``example_ids``.
            metadata: A dictionary of metadata.

        Returns:
            The prediction result.
        """
        if len(example_ids) != len(predictions):
            msg = "example_ids and predictions must have the same length"
            raise ValueError(msg)
        records = [
            PredictionRecord(example_id=example_id, prediction=prediction)
            for example_id, prediction in zip(example_ids, predictions)
        ]
        return PredictionResult(records=records, metadata=metadata)

    @classmethod
    def from_dict(
        cls, data: dict[str, T], metadata: dict[str, Any] | None = None
    ) -> PredictionResult[T]:
        r"""Create a prediction result from a dictionary.

        Args:
            data: A dictionary containing example IDs and predictions.
            metadata: A dictionary of metadata.

        Returns:
            The prediction result.
        """
        return cls.from_predictions(
            example_ids=data.keys(),
            predictions=data.values(),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, T]:
        r"""Return a dictionary containing example IDs and predictions.

        Returns:
            A dictionary containing example IDs and predictions.
        """
        return {record.example_id: record.prediction for record in self.records}
