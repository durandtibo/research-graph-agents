r"""Contain code to implement inference pipelines."""

from __future__ import annotations

__all__ = ["BaseInferencePipeline", "InferencePipeline"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    from argos.tasks.autoprompt.predictor import BasePredictor


logger: logging.Logger = logging.getLogger(__name__)


class BaseInferencePipeline(ABC):
    r"""Define the base class to implement inference pipeline."""

    @abstractmethod
    def process(self) -> pl.DataFrame:
        r"""Process the results of the inference pipeline.

        Returns:
            The result of the inference pipeline.
        """


class InferencePipeline(BaseInferencePipeline):
    r"""Implement a simple inference pipeline."""

    def __init__(
        self,
        dataset: pl.DataFrame,
        predictor: BasePredictor,
        path_results: Path | None = None,
    ) -> None:
        self._dataset = dataset
        self._predictor = predictor
        self._path_results = path_results

    def process(self) -> pl.DataFrame:
        if self._path_results and self._path_results.is_file():
            logger.info(f"Reading results from {self._path_results}...")
            return pl.read_parquet(self._path_results)

        results = self._predictor.predict(self._dataset)
        if self._path_results:
            logger.info(f"Writing results ({results.shape}) in {self._path_results}...")
            self._path_results.parent.mkdir(parents=True, exist_ok=True)
            results.write_parquet(self._path_results)
        return results
