r"""Provide inference pipelines for the haiku autoprompt workflow."""

from __future__ import annotations

__all__ = ["BaseInferencePipeline", "InferencePipeline"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import polars as pl
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from pathlib import Path

    from argos.autoprompt.haiku.predictor import BasePredictor


logger: logging.Logger = logging.getLogger(__name__)


class BaseInferencePipeline(ABC):
    r"""Abstract base class for inference pipelines.

    Subclasses must implement :meth:`process` to run the end-to-end
    prediction step and return the results as a
    :class:`~polars.DataFrame`.
    """

    @abstractmethod
    def process(self) -> pl.DataFrame:
        r"""Run the inference pipeline and return the predictions.

        Returns:
            A :class:`~polars.DataFrame` containing the prediction
                results for the entire dataset.
        """


class InferencePipeline(BaseInferencePipeline):
    r"""Implement a simple inference pipeline.

    Args:
        dataset: The dataset to use for inference.
        predictor: The predictor used to generate predictions.
        path: Optional path for caching predictions as a Parquet
            file. If the file already exists, predictions are read
            from it instead of running inference. If ``None``, no
            caching is performed.
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
        args = repr_indent(repr_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def process(self) -> pl.DataFrame:
        r"""Run inference and optionally reuse or populate a cache file.

        If ``path`` was provided and already points to an existing file,
        predictions are loaded from that Parquet file and the predictor is
        not executed. Otherwise, predictions are generated from
        ``dataset`` and optionally written to ``path``.

        Returns:
            A :class:`~polars.DataFrame` containing prediction results.

        Example:
            ```pycon
            >>> import polars as pl
            >>> from langchain_core.runnables import RunnableLambda
            >>> from argos.autoprompt.haiku.inference import InferencePipeline
            >>> from argos.autoprompt.haiku.predictor import Predictor
            >>> dataset = pl.DataFrame({"topic": ["ocean"], "haiku": ["waves break softly"]})
            >>> model = RunnableLambda(lambda row: {"overall_prediction": 1})
            >>> predictor = Predictor(model=model, batch_size=1)
            >>> pipeline = InferencePipeline(dataset=dataset, predictor=predictor, path=None)
            >>> "overall_prediction" in pipeline.process().columns
            True

            ```
        """
        if self._path and self._path.is_file():
            logger.info(f"Reading predictions from {self._path}...")
            return pl.read_parquet(self._path)

        predictions = self._predictor.predict(self._dataset)
        if self._path:
            logger.info(f"Writing predictions (shape={predictions.shape}) to {self._path}...")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            predictions.write_parquet(self._path)
        return predictions

    def _get_kwargs(self) -> dict[str, Any]:
        return {
            "dataset": f"shape: {self._dataset.shape}",
            "predictor": self._predictor,
            "path": self._path,
        }
