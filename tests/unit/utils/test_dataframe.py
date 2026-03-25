from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from argos.utils.dataframe import concat_and_merge, summarize_boolean_columns

######################################
#     Tests for concat_and_merge     #
######################################


def test_concat_and_merge_no_shared_columns_returns_all_columns() -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"c": [5, 6], "d": [7, 8]})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(result, pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]}))


def test_concat_and_merge_no_shared_columns_preserves_values() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [3, 4]})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(result, pl.DataFrame({"a": [1, 2], "b": [3, 4]}))


def test_concat_and_merge_shared_column_df1_value_wins_over_df2() -> None:
    df1 = pl.DataFrame({"a": [1]})
    df2 = pl.DataFrame({"a": [99]})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(result, pl.DataFrame({"a": [1]}))


def test_concat_and_merge_shared_column_null_in_df1_falls_back_to_df2() -> None:
    df1 = pl.DataFrame({"a": [None], "b": [1]})
    df2 = pl.DataFrame({"a": [99]})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(result, pl.DataFrame({"a": [99], "b": [1]}))


def test_concat_and_merge_shared_column_both_null_stays_null() -> None:
    df1 = pl.DataFrame({"a": pl.Series([None], dtype=pl.Int64)})
    df2 = pl.DataFrame({"a": pl.Series([None], dtype=pl.Int64)})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(result, pl.DataFrame({"a": pl.Series([None], dtype=pl.Int64)}))


def test_concat_and_merge_shared_column_mixed_nulls_across_rows() -> None:
    df1 = pl.DataFrame({"x": [1, None, None]})
    df2 = pl.DataFrame({"x": [None, 2, None]})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(result, pl.DataFrame({"x": [1, 2, None]}))


def test_concat_and_merge_shared_column_no_temp_suffix_column_leaks() -> None:
    """Temporary rename columns must not appear in the output."""
    df1 = pl.DataFrame({"a": [1]})
    df2 = pl.DataFrame({"a": [2]})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(result, pl.DataFrame({"a": [1]}))


def test_concat_and_merge_multiple_shared_columns_all_coalesced() -> None:
    df1 = pl.DataFrame({"a": [1, None], "b": [None, 20]})
    df2 = pl.DataFrame({"a": [9, 2], "b": [10, 99]})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(result, pl.DataFrame({"a": [1, 2], "b": [10, 20]}))


def test_concat_and_merge_mixed_shared_and_unique_columns() -> None:
    df1 = pl.DataFrame({"a": [1, None], "only1": [10, 20]})
    df2 = pl.DataFrame({"a": [9, 2], "only2": [30, 40]})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(result, pl.DataFrame({"a": [1, 2], "only1": [10, 20], "only2": [30, 40]}))


def test_column_order_preserves_df1_columns_first() -> None:
    df1 = pl.DataFrame({"a": [1], "b": [2]})
    df2 = pl.DataFrame({"b": [2], "c": [3]})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(result, pl.DataFrame({"a": [1], "b": [2], "c": [3]}))


def test_concat_and_merge_both_empty_dataframes_returns_empty() -> None:
    df1 = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    df2 = pl.DataFrame({"b": pl.Series([], dtype=pl.Int64)})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(
        result,
        pl.DataFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.Int64)}),
    )


def test_concat_and_merge_shared_column_empty_dataframes_returns_empty() -> None:
    df1 = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    df2 = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    result = concat_and_merge(df1, df2)
    assert_frame_equal(result, pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}))


def test_concat_and_merge_row_count_mismatch_raises() -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"b": [1, 2]})
    with pytest.raises(ValueError, match="DataFrames must have the same number of rows"):
        concat_and_merge(df1, df2)


###############################################
#     Tests for summarize_boolean_columns     #
###############################################


def test_summarize_boolean_columns_all_true() -> None:
    df = pl.DataFrame({"a": [True, True, True]})
    result = summarize_boolean_columns(df)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "column": ["a"],
                "true_count": [3],
                "false_count": [0],
                "true_pct": [100.0],
                "false_pct": [0.0],
            }
        ),
    )


def test_summarize_boolean_columns_all_false() -> None:
    df = pl.DataFrame({"a": [False, False, False]})
    result = summarize_boolean_columns(df)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "column": ["a"],
                "true_count": [0],
                "false_count": [3],
                "true_pct": [0.0],
                "false_pct": [100.0],
            }
        ),
    )


def test_summarize_boolean_columns_mixed_true_and_false() -> None:
    df = pl.DataFrame({"a": [True, True, False, True]})
    result = summarize_boolean_columns(df)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "column": ["a"],
                "true_count": [3],
                "false_count": [1],
                "true_pct": [75.0],
                "false_pct": [25.0],
            }
        ),
    )


def test_summarize_boolean_columns_single_row_true() -> None:
    df = pl.DataFrame({"a": [True]})
    result = summarize_boolean_columns(df)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "column": ["a"],
                "true_count": [1],
                "false_count": [0],
                "true_pct": [100.0],
                "false_pct": [0.0],
            }
        ),
    )


def test_summarize_boolean_columns_single_row_false() -> None:
    df = pl.DataFrame({"a": [False]})
    result = summarize_boolean_columns(df)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "column": ["a"],
                "true_count": [0],
                "false_count": [1],
                "true_pct": [0.0],
                "false_pct": [100.0],
            }
        ),
    )


def test_summarize_boolean_columns_multiple_columns() -> None:
    df = pl.DataFrame(
        {
            "is_active": [True, True, False, True],
            "has_error": [False, False, False, True],
        }
    )
    result = summarize_boolean_columns(df)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "column": ["is_active", "has_error"],
                "true_count": [3, 1],
                "false_count": [1, 3],
                "true_pct": [75.0, 25.0],
                "false_pct": [25.0, 75.0],
            }
        ),
    )


def test_summarize_boolean_columns_row_order_matches_input_column_order() -> None:
    df = pl.DataFrame({"z": [True], "a": [False], "m": [True]})
    result = summarize_boolean_columns(df)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "column": ["z", "a", "m"],
                "true_count": [1, 0, 1],
                "false_count": [0, 1, 0],
                "true_pct": [100.0, 0.0, 100.0],
                "false_pct": [0.0, 100.0, 0.0],
            }
        ),
    )


def test_summarize_boolean_columns_raises_on_integer_column() -> None:
    df = pl.DataFrame({"a": [1, 0, 1]})
    with pytest.raises(ValueError, match="Non-boolean columns"):
        summarize_boolean_columns(df)


def test_summarize_boolean_columns_raises_on_string_column() -> None:
    df = pl.DataFrame({"a": ["true", "false"]})
    with pytest.raises(ValueError, match="Non-boolean columns"):
        summarize_boolean_columns(df)


def test_summarize_boolean_columns_raises_on_mixed_bool_and_non_bool() -> None:
    df = pl.DataFrame({"a": [True, False], "b": [1.0, 0.0]})

    with pytest.raises(ValueError, match="Non-boolean columns"):
        summarize_boolean_columns(df)
