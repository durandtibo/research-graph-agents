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

    Example:
        ```pycon
        >>> from argos.meta_agent.prediction import PredictionRecord
        >>> record = PredictionRecord(example_id="q1", prediction="4")
        >>> record.example_id
        'q1'
        >>> record.prediction
        '4'

        ```
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

    Example:
        ```pycon
        >>> from argos.meta_agent.prediction import PredictionRecord, PredictionResult
        >>> result = PredictionResult(
        ...     records=[
        ...         PredictionRecord(example_id="q1", prediction="4"),
        ...         PredictionRecord(example_id="q2", prediction="6"),
        ...     ]
        ... )
        >>> result.to_dict()
        {'q1': '4', 'q2': '6'}

        ```
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
        r"""Create a prediction result from a list of example IDs and
        predictions.

        ``example_ids`` and ``predictions`` must be aligned, i.e. the
        n-th example ID must match the n-th prediction.

        Args:
            example_ids: A list of example IDs.
            predictions: A list of predictions aligned with
                ``example_ids``.
            metadata: A dictionary of metadata.

        Returns:
            The prediction result.

        Raises:
            ValueError: If ``example_ids`` and ``predictions`` have
                different lengths.

        Example:
            ```pycon
            >>> from argos.meta_agent.prediction import PredictionResult
            >>> result = PredictionResult.from_predictions(
            ...     example_ids=["q1", "q2"],
            ...     predictions=["4", "6"],
            ... )
            >>> result.to_dict()
            {'q1': '4', 'q2': '6'}

            ```
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
            data: A dictionary mapping example IDs to predictions.
            metadata: A dictionary of metadata.

        Returns:
            The prediction result.

        Example:
            ```pycon
            >>> from argos.meta_agent.prediction import PredictionResult
            >>> result = PredictionResult.from_dict({"q1": "4", "q2": "6"})
            >>> result.to_dict()
            {'q1': '4', 'q2': '6'}

            ```
        """
        return cls.from_predictions(
            example_ids=data.keys(),
            predictions=data.values(),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, T]:
        r"""Return a dictionary mapping example IDs to predictions.

        The result only includes the prediction payload. Record-level and
        result-level metadata are omitted.

        Returns:
            A dictionary mapping each example ID to its prediction.

        Example:
            ```pycon
            >>> from argos.meta_agent.prediction import PredictionRecord, PredictionResult
            >>> result = PredictionResult(
            ...     records=[
            ...         PredictionRecord(example_id="q1", prediction="4"),
            ...         PredictionRecord(example_id="q2", prediction="6"),
            ...     ]
            ... )
            >>> result.to_dict()
            {'q1': '4', 'q2': '6'}

            ```
        """
        return {record.example_id: record.prediction for record in self.records}
