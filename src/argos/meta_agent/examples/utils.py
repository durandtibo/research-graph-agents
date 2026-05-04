r"""Contain utilities for examples."""

from __future__ import annotations

__all__ = ["dataframe_to_examples", "examples_to_dataframe"]

from typing import TypeVar

import polars as pl

from argos.meta_agent.examples import BaseExample
from argos.meta_agent.examples.vanilla import Example

ExampleT = TypeVar("ExampleT", bound=BaseExample)


def dataframe_to_examples(
    frame: pl.DataFrame, example_type: type[ExampleT] = Example
) -> list[ExampleT]:
    r"""Convert a Polars DataFrame into a list of examples.

    Each row in the DataFrame is converted into a single example using
    the ``from_dict`` class method of ``example_type``. This is the
    inverse of :func:`examples_to_dataframe`.

    Args:
        frame: A Polars DataFrame where each row represents a single
            example. Column names must match the fields expected by
            ``example_type.from_dict``.
        example_type: The example class to instantiate for each row.
            Defaults to :class:`Example`.

    Returns:
        A list of examples, one per row, in the same order as the
            DataFrame.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.meta_agent.examples import Example, dataframe_to_examples
        >>> frame = pl.DataFrame(
        ...     {
        ...         "id": ["q1", "q2"],
        ...         "input": ["What is 2+2?", "What is 4+2?"],
        ...         "target": ["4", "6"],
        ...         "metadata": [None, None],
        ...     }
        ... )
        >>> examples = dataframe_to_examples(frame)
        >>> examples[0]
        Example(id='q1', input='What is 2+2?', target='4', metadata=None)

        ```
    """
    return [example_type.from_dict(row) for row in frame.iter_rows(named=True)]


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
