r"""Contain code to generate predictions."""

from __future__ import annotations

__all__ = ["BasePredictor", "Predictor", "generate_predictions"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import polars as pl
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping
from coola.utils.timing import timeblock
from langchain_core.runnables import Runnable, RunnableConfig

from argos.utils.batching import batchify
from argos.utils.dataframe import concat_and_merge, unnest_struct_columns
from argos.utils.mapping import recursive_to_dict

if TYPE_CHECKING:
    from collections.abc import Sequence

logger: logging.Logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    r"""Define the base class to generate the predictions."""

    @abstractmethod
    def predict(self, dataset: pl.DataFrame) -> pl.DataFrame:
        r"""Compute the predictions for the given dataset.

        Args:
            dataset: The dataset to predict on.

        Returns:
            The predictions for the given dataset.
        """


class Predictor(BasePredictor):
    r"""Implement a simple predictor.

    Args:
        model: The :class:`~langchain_core.runnables.Runnable` used
            to generate predictions.
        batch_size: Number of inputs to process concurrently per
            batch. Defaults to ``20``.
        output_columns: Column names to include in the returned
            :class:`~polars.DataFrame`. If ``None``, all columns
            are returned.
        config: Optional
            :class:`~langchain_core.runnables.RunnableConfig`
            controlling concurrency. If ``None``, defaults to
            ``RunnableConfig(max_concurrency=batch_size)``.
    """

    def __init__(
        self,
        model: Runnable[Any, Any],
        batch_size: int = 20,
        output_columns: Sequence[str] | None = None,
        config: RunnableConfig | None = None,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._output_columns = output_columns
        self._config = config or RunnableConfig(max_concurrency=batch_size)

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def predict(self, dataset: pl.DataFrame) -> pl.DataFrame:
        predictions = generate_predictions(
            model=self._model,
            dataset=dataset,
            batch_size=self._batch_size,
            config=self._config,
        )
        predictions = predictions.select(sorted(predictions.columns))
        if self._output_columns is not None:
            predictions = predictions.select(self._output_columns)
        return predictions

    def _get_kwargs(self) -> dict[str, Any]:
        return {
            "model": self._model,
            "batch_size": self._batch_size,
            "output_columns": self._output_columns,
            "config": self._config,
        }


def generate_predictions(
    model: Runnable[Any, Any],
    dataset: pl.DataFrame,
    batch_size: int = 20,
    config: RunnableConfig | None = None,
) -> pl.DataFrame:
    r"""Run the inference and returns the results in a DataFrame.

    Args:
        model: The :class:`~langchain_core.runnables.Runnable` used
            to generate the predictions.
        dataset: The dataset to run inference on.
        batch_size: Number of inputs to process concurrently per
            batch. Defaults to ``20``.
        config: Optional
            :class:`~langchain_core.runnables.RunnableConfig`
            controlling concurrency. If ``None``, a default config
            with ``max_concurrency=batch_size`` is used.

    Returns:
        The results of the inference.
    """
    logger.info(f"Running inference with {batch_size:,} batches...")
    batches = batchify(list(dataset.iter_rows(named=True)), size=batch_size)

    outputs = []
    with timeblock(message="LLM inference time: {time}"):
        for index, batch in enumerate(batches):
            logger.info(f"--- Processing Batch {index + 1} ---")
            outputs.extend(model.batch(batch, config=config))

    logger.info("Preparing predictions...")
    predictions = pl.from_dicts(recursive_to_dict(outputs))
    return concat_and_merge(dataset, unnest_struct_columns(predictions))
