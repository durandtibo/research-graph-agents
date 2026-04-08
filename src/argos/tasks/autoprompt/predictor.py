r"""Contain code to generate predictions."""

from __future__ import annotations

__all__ = ["BasePredictor", "Predictor", "generate_predictions", "prepare_results"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import polars as pl
from coola.utils.format import repr_indent, repr_mapping
from coola.utils.timing import timeblock

from argos.utils.batching import batchify
from argos.utils.dataframe import concat_and_merge

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


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
        graph: The graph of the haiku judge.
        batch_size: The batch size for inference.
    """

    def __init__(self, graph: CompiledStateGraph, batch_size: int = 20) -> None:
        self._graph = graph
        self._batch_size = batch_size

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"graph": self._graph, "batch_size": self._batch_size}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def predict(self, dataset: pl.DataFrame) -> pl.DataFrame:
        r"""Compute the predictions for the given dataset.

        Args:
            dataset: The dataset to predict on.

        Returns:
            A :class:`~polars.DataFrame` containing the predictions
                for each row in ``dataset``.
        """
        return generate_predictions(dataset=dataset, graph=self._graph, batch_size=self._batch_size)


def generate_predictions(
    dataset: pl.DataFrame, graph: CompiledStateGraph, batch_size: int = 20
) -> pl.DataFrame:
    r"""Run the inference and returns the results in a DataFrame.

    Args:
        dataset: The dataset to run inference on.
        graph: The graph of the haiku judge.
        batch_size: The batch size for inference.

    Returns:
        The results of the inference.
    """
    logger.info(f"Running inference with {batch_size:,} batches...")
    outputs = []
    examples = list(dataset.iter_rows(named=True))
    batches = batchify(examples, size=batch_size)

    with timeblock(message="LLM inference time: {time}"):
        for index, batch in enumerate(batches):
            logger.info(f"--- Processing Batch {index + 1} ---")
            output = graph.batch(batch, config={"max_concurrency": batch_size})
            outputs.extend(output)

    logger.info("Preparing results...")
    return prepare_results(dataset, outputs)


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
