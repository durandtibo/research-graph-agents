r"""Contain code to generate predictions."""

from __future__ import annotations

__all__ = ["BasePredictor", "Predictor", "generate_predictions", "prepare_results"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import polars as pl
from coola.utils.format import repr_indent, repr_mapping
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
        model: The model.
        batch_size: The batch size for inference.
        output_columns: Select the columns to return. If ``None``, all the columns are returned.
        config: A runnable config.
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
        args = repr_indent(
            repr_mapping(
                {
                    "model": self._model,
                    "batch_size": self._batch_size,
                    "output_columns": self._output_columns,
                    "config": self._config,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def predict(self, dataset: pl.DataFrame) -> pl.DataFrame:
        predictions = generate_predictions(
            model=self._model,
            dataset=dataset,
            batch_size=self._batch_size,
            config=self._config,
        )
        if self._output_columns is not None:
            predictions = predictions.select(self._output_columns)
        return predictions


def generate_predictions(
    model: Runnable[Any, Any],
    dataset: pl.DataFrame,
    batch_size: int = 20,
    config: RunnableConfig | None = None,
) -> pl.DataFrame:
    r"""Run the inference and returns the results in a DataFrame.

    Args:
        dataset: The dataset to run inference on.
        model: The model used to generate the predictions.
        batch_size: The batch size for inference.
        config: A runnable config.

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


def prepare_results(dataset: pl.DataFrame, outputs: list[dict[Any, Any]]) -> pl.DataFrame:
    r"""Prepare results of haiku judge.

    Args:
        dataset: The dataset of haiku examples.
        outputs: The results of the haiku judge.

    Returns:
        The results of the haiku judge in a DataFrame.
    """
    cols = [
        "topic",
        "haiku",
        "score",
        "passed",
        "target",
        "structure_passed",
        "structure_target",
        "topic_passed",
        "topic_target",
        "reasoning",
    ]
    flat_data = [
        {**{k: v for k, v in row.items() if k != "evaluation"}, **row["evaluation"].model_dump()}
        for row in outputs
    ]
    return concat_and_merge(pl.DataFrame(flat_data), dataset).select(cols)
