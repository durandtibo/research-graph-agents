r"""Contain code to implement inference pipelines."""

from __future__ import annotations

__all__ = ["BaseInferencePipeline", "InferencePipeline"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import polars as pl
from coola.utils.format import repr_indent, repr_mapping

if TYPE_CHECKING:
    from pathlib import Path

    from argos.tasks.autoprompt.predictor import BasePredictor


logger: logging.Logger = logging.getLogger(__name__)


class BaseInferencePipeline(ABC):
    r"""Define the base class to implement inference pipeline."""

    @abstractmethod
    def process(self) -> pl.DataFrame:
        r"""Process the predictions of the inference pipeline.

        Returns:
            The result of the inference pipeline.
        """


class InferencePipeline(BaseInferencePipeline):
    r"""Implement a simple inference pipeline.

    Args:
        dataset: The dataset to use for inference.
        predictor: The predictor to use for inference.
        path: Path where to read/write the predictions.
    """

    def __init__(
        self,
        dataset: pl.DataFrame,
        predictor: BasePredictor,
        path: Path | None = None,
    ) -> None:
        self._dataset = dataset
        self._predictor = predictor
        self._path = path

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "dataset": f"shape: {self._dataset.shape}",
                    "predictor": self._predictor,
                    "path": self._path,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def process(self) -> pl.DataFrame:
        r"""Run inference and return the results.

        If a predictions file already exists at ``path``, it is loaded
        and returned directly without re-running the predictor.
        Otherwise, the predictor is invoked on the dataset, the results
        are written to ``path`` (if provided), and then returned.

        Returns:
            A :class:`~polars.DataFrame` containing the inference
                results produced by the predictor.
        """
            logger.info(f"Reading predictions from {self._path}...")
            return pl.read_parquet(self._path)

        predictions = self._predictor.predict(self._dataset)
        if self._path:
            logger.info(f"Writing predictions ({predictions.shape}) in {self._path}...")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            predictions.write_parquet(self._path)
        return predictions
