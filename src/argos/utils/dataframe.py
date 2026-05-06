r"""Utility functions for processing Polars DataFrames."""

from __future__ import annotations

from io import StringIO
from typing import Any

import polars as pl

__all__ = [
    "concat_and_merge",
    "dataframe_to_csv",
    "dataframe_to_markdown",
    "list_of_dicts_to_dataframe",
    "summarize_boolean_columns",
    "unnest_struct_columns",
]

from argos.utils.mapping import recursive_to_dict

_TEMP_SUFFIX = "_temp_right"


def concat_and_merge(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    r"""Horizontally concatenate two DataFrames, coalescing any shared
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
        ValueError: If ``df1`` and ``df2`` have a different number of rows.

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


def list_of_dicts_to_dataframe(list_of_dicts: list[dict[Any, Any]]) -> pl.DataFrame:
    r"""Convert a list of dicts into a DataFrame.

    Nested objects in each dict are recursively converted to plain dicts
    before the DataFrame is constructed.

    Args:
        list_of_dicts: A list of dicts where each dict represents a
            single row, with keys as column names and values as the
            corresponding cell values.

    Returns:
        A DataFrame representation of the list of dicts.
    """
    rows = recursive_to_dict(list_of_dicts)
    return pl.DataFrame(rows)


def summarize_boolean_columns(df: pl.DataFrame) -> pl.DataFrame:
    r"""Summarize the count of True and False values for each boolean
    column.

    Each row in the output represents one column from the input DataFrame,
    with the counts of True and False values and their respective percentages.

    Args:
        df: A DataFrame whose columns are all boolean.

    Returns:
        A DataFrame with one row per input column and the following columns:

        - ``column``: the name of the original column.
        - ``true_count``: number of True values.
        - ``false_count``: number of False values.
        - ``true_pct``: percentage of True values (0-100).
        - ``false_pct``: percentage of False values (0-100).

    Raises:
        ValueError: If any column in ``df`` is not of boolean dtype.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.utils.dataframe import summarize_boolean_columns
        >>> df = pl.DataFrame(
        ...     {
        ...         "is_active": [True, True, False, True],
        ...         "has_error": [False, False, False, True],
        ...     }
        ... )
        >>> summarize_boolean_columns(df)
        shape: (2, 5)
        ┌───────────┬────────────┬─────────────┬──────────┬───────────┐
        │ column    ┆ true_count ┆ false_count ┆ true_pct ┆ false_pct │
        │ ---       ┆ ---        ┆ ---         ┆ ---      ┆ ---       │
        │ str       ┆ i64        ┆ i64         ┆ f64      ┆ f64       │
        ╞═══════════╪════════════╪═════════════╪══════════╪═══════════╡
        │ is_active ┆ 3          ┆ 1           ┆ 75.0     ┆ 25.0      │
        │ has_error ┆ 1          ┆ 3           ┆ 25.0     ┆ 75.0      │
        └───────────┴────────────┴─────────────┴──────────┴───────────┘

        ```
    """
    non_bool = [col for col in df.columns if df[col].dtype != pl.Boolean]
    if non_bool:
        msg = f"All columns must be boolean. Non-boolean columns: {non_bool}"
        raise ValueError(msg)

    n_rows = df.height
    return pl.DataFrame(
        {
            "column": df.columns,
            "true_count": [df[col].sum() for col in df.columns],
            "false_count": [n_rows - df[col].sum() for col in df.columns],
        }
    ).with_columns(
        [
            (pl.col("true_count") / n_rows * 100).alias("true_pct"),
            (pl.col("false_count") / n_rows * 100).alias("false_pct"),
        ]
    )


def unnest_struct_columns(frame: pl.DataFrame, separator: str | None = None) -> pl.DataFrame:
    r"""Unnest all the struct columns in a DataFrame.

    Args:
        frame: A DataFrame that can have struct columns.
        separator: If provided, renamed output columns follow the
            ``<struct_col><separator><field>`` pattern. If ``None``,
            the original field names are kept as-is.

    Returns:
        A DataFrame where all the struct columns are unnested.
    """
    struct_columns = [
        col_name for col_name, dtype in frame.schema.items() if isinstance(dtype, pl.Struct)
    ]

    return frame.unnest(struct_columns, separator=separator)


def dataframe_to_csv(frame: pl.DataFrame) -> str:
    r"""Return a CSV string representation of the DataFrame.

    The returned string starts with a ``Schema:`` line listing every
    column name and its dtype, followed by the CSV body. If the frame
    is empty the string ``"_No data available._"`` is returned instead.

    Args:
        frame: A DataFrame.

    Returns:
        A CSV string (with a schema header) representing the
            DataFrame, or ``"_No data available._"`` when the frame
            is empty.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.utils.dataframe import dataframe_to_csv
        >>> frame = pl.DataFrame({"loss": [0.5, 0.3], "accuracy": [0.9, 0.95]})
        >>> print(dataframe_to_csv(frame))
        Schema: loss: Float64, accuracy: Float64
        <BLANKLINE>
        loss,accuracy
        0.5,0.9
        0.3,0.95
        <BLANKLINE>

        ```
    """
    if frame.is_empty():
        return "_No data available._"
    schema = ", ".join(f"{name}: {dtype}" for name, dtype in frame.schema.items())
    buf = StringIO()
    frame.write_csv(buf)
    return f"Schema: {schema}\n\n{buf.getvalue()}"


def dataframe_to_markdown(frame: pl.DataFrame) -> str:
    r"""Return a markdown string representation of the DataFrame.

    Builds a GitHub-flavored markdown table from the DataFrame. If the
    frame is empty, ``"_No data available._"`` is returned instead.

    Args:
        frame: A DataFrame.

    Returns:
        A markdown table string representing the DataFrame, or
            ``"_No data available._"`` when the frame is empty.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.utils.dataframe import dataframe_to_markdown
        >>> frame = pl.DataFrame({"loss": [0.5, 0.3], "accuracy": [0.9, 0.95]})
        >>> print(dataframe_to_markdown(frame))
        | loss | accuracy |
        |------|----------|
        | 0.5  | 0.9      |
        | 0.3  | 0.95     |

        ```
    """
    if frame.is_empty():
        return "_No data available._"
    col_widths = {col: max(len(col), *(len(str(v)) for v in frame[col])) for col in frame.columns}

    def fmt(value: object, col: str) -> str:
        return str(value).ljust(col_widths[col])

    header = "| " + " | ".join(col.ljust(col_widths[col]) for col in frame.columns) + " |"
    separator = "|" + "|".join("-" * (col_widths[col] + 2) for col in frame.columns) + "|"
    rows = [
        "| " + " | ".join(fmt(row[col], col) for col in frame.columns) + " |"
        for row in frame.iter_rows(named=True)
    ]
    return "\n".join([header, separator, *rows])
