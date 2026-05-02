r"""Contain utilities for results."""

from __future__ import annotations

__all__ = ["results_to_dataframe"]

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from argos.meta_agent.results import BaseResult


def results_to_dataframe(results: list[BaseResult]) -> pl.DataFrame:
    r"""Convert a list of results into a Polars DataFrame.

    Each result is represented as a single row, where columns correspond
    to the metric names returned by ``to_flat_dict``. This is the
    recommended way to build a DataFrame from multiple results, as a
    single result does not carry enough structure to justify a DataFrame
    on its own.

    Args:
        results: A list of result objects to convert. All results should
            have consistent keys in their ``to_flat_dict`` output to
            ensure a well-formed DataFrame. An empty list returns an
            empty DataFrame.

    Returns:
        A Polars DataFrame with one row per result and one column per
            metric.

    Example:
        ```pycon
        >>> from argos.meta_agent.results import Result
        >>> from argos.meta_agent.results.utils import results_to_dataframe
        >>> df = results_to_dataframe(
        ...     [Result({"loss": 0.5, "accuracy": 0.9}), Result({"loss": 0.3, "accuracy": 0.95})]
        ... )
        >>> df.shape
        (2, 2)

        ```
    """
    return pl.DataFrame([r.to_flat_dict() for r in results])
