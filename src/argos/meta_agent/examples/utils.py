r"""Contain utilities for examples."""

from __future__ import annotations

__all__ = ["examples_to_dataframe"]

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from argos.meta_agent.examples import BaseExample


def examples_to_dataframe(examples: list[BaseExample]) -> pl.DataFrame:
    r"""Convert a list of examples into a Polars DataFrame.

    Each example is represented as a single row, where columns correspond
    to the fields returned by ``to_dict``. This is the recommended way to
    build a DataFrame from multiple examples, as a single example does not
    carry enough structure to justify a DataFrame on its own.

    Args:
        examples: A list of examples to convert. All examples should have
            consistent keys in their ``to_dict`` output to ensure a
            well-formed DataFrame. An empty list returns an empty DataFrame.

    Returns:
        A Polars DataFrame with one row per example and one column per
            field.

    Example:
        ```pycon
        >>> from argos.meta_agent.examples import Example, examples_to_dataframe
        >>> frame = examples_to_dataframe(
        ...     [
        ...         Example(id="q1", input="What is 2+2?", target="4"),
        ...         Example(id="q2", input="What is 4+2?", target="6"),
        ...     ]
        ... )
        >>> frame
        shape: (2, 4)
        ┌─────┬──────────────┬────────┬──────────┐
        │ id  ┆ input        ┆ target ┆ metadata │
        │ --- ┆ ---          ┆ ---    ┆ ---      │
        │ str ┆ str          ┆ str    ┆ null     │
        ╞═════╪══════════════╪════════╪══════════╡
        │ q1  ┆ What is 2+2? ┆ 4      ┆ null     │
        │ q2  ┆ What is 4+2? ┆ 6      ┆ null     │
        └─────┴──────────────┴────────┴──────────┘

        ```
    """
    return pl.DataFrame([r.to_dict() for r in examples])
