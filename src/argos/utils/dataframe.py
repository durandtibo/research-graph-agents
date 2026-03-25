r"""Utility functions for processing Polars DataFrames."""

from __future__ import annotations

import polars as pl

__all__ = ["concat_and_merge"]

_TEMP_SUFFIX = "_temp_right"


def concat_and_merge(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    """Horizontally concatenate two DataFrames, coalescing any shared
    columns.

    Columns that exist only in one DataFrame are included as-is. Columns that
    appear in both are merged by taking the first non-null value from ``df1``,
    falling back to the value in ``df2`` (i.e. ``COALESCE(df1.col, df2.col)``).

    Args:
        df1: Left DataFrame. Its values take priority during coalescing.
        df2: Right DataFrame. Its values are used as fallback during coalescing.

    Returns:
        A new DataFrame with the same number of rows as the inputs and a column
        set equal to the union of both column sets.

    Raises:
        ShapeError: If ``df1`` and ``df2`` have a different number of rows.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.utils.dataframe import concat_and_merge
        >>> df1 = pl.DataFrame({"a": [1, 2], "b": [10, 20]})
        >>> df2 = pl.DataFrame({"a": [1, 2], "c": [30, 40]})
        >>> concat_and_merge(df1, df2)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 10  ┆ 30  │
        │ 2   ┆ 20  ┆ 40  │
        └─────┴─────┴─────┘

        ```
    """
    if df1.height != df2.height:
        msg = f"DataFrames must have the same number of rows, got {df1.height} and {df2.height}."
        raise ValueError(msg)

    shared_cols = [col for col in df1.columns if col in df2.columns]

    if not shared_cols:
        return pl.concat([df1, df2], how="horizontal")

    df2_renamed = df2.rename({col: f"{col}{_TEMP_SUFFIX}" for col in shared_cols})
    combined = pl.concat([df1, df2_renamed], how="horizontal")

    coalesced = [pl.coalesce(col, f"{col}{_TEMP_SUFFIX}").alias(col) for col in shared_cols]
    return combined.with_columns(coalesced).drop([f"{col}{_TEMP_SUFFIX}" for col in shared_cols])
