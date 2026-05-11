r"""Contain utilities for examples."""

from __future__ import annotations

__all__ = ["dataframe_to_examples", "examples_to_dataframe"]

from typing import TYPE_CHECKING, TypeVar

import polars as pl

from argos.meta_agent.examples import BaseExample
from argos.meta_agent.examples.vanilla import Example

if TYPE_CHECKING:
    from argos.meta_agent.typing import InputT, TargetT

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


def examples_to_dataframe(
    examples: list[BaseExample[InputT, TargetT]],
    *,
    unnest_columns: bool = False,
) -> pl.DataFrame:
    r"""Convert a list of examples into a Polars DataFrame.

    Each example is represented as a single row, where columns correspond
    to the fields returned by ``to_dict``. This is the recommended way to
    build a DataFrame from multiple examples, as a single example does not
    carry enough structure to justify a DataFrame on its own.

    Args:
        examples: A list of examples to convert. All examples should have
            consistent keys in their ``to_dict`` output to ensure a
            well-formed DataFrame. An empty list returns an empty DataFrame.
        unnest_columns: If ``True``, nested fields are flattened into
            separate top-level columns using ``to_flat_dict`` (e.g.
            ``metadata.score`` becomes its own column). If ``False``
            (default), nested fields are kept as struct columns.

    Returns:
        A Polars DataFrame with one row per example and one column per
            field.

    Example:
        ```pycon
        >>> from argos.meta_agent.examples import Example, examples_to_dataframe
        >>> examples = [
        ...     Example(id="q1", input="What is 2+2?", target="4", metadata={"source": "cat"}),
        ...     Example(id="q2", input="What is 4+2?", target="6", metadata={"source": "bear"}),
        ... ]
        >>> frame = examples_to_dataframe(examples)
        >>> frame
        shape: (2, 4)
        ┌─────┬──────────────┬────────┬───────────┐
        │ id  ┆ input        ┆ target ┆ metadata  │
        │ --- ┆ ---          ┆ ---    ┆ ---       │
        │ str ┆ str          ┆ str    ┆ struct[1] │
        ╞═════╪══════════════╪════════╪═══════════╡
        │ q1  ┆ What is 2+2? ┆ 4      ┆ {"cat"}   │
        │ q2  ┆ What is 4+2? ┆ 6      ┆ {"bear"}  │
        └─────┴──────────────┴────────┴───────────┘
        >>> frame = examples_to_dataframe(examples, unnest_columns=True)
        >>> frame
        shape: (2, 4)
        ┌─────┬──────────────┬────────┬─────────────────┐
        │ id  ┆ input        ┆ target ┆ metadata.source │
        │ --- ┆ ---          ┆ ---    ┆ ---             │
        │ str ┆ str          ┆ str    ┆ str             │
        ╞═════╪══════════════╪════════╪═════════════════╡
        │ q1  ┆ What is 2+2? ┆ 4      ┆ cat             │
        │ q2  ┆ What is 4+2? ┆ 6      ┆ bear            │
        └─────┴──────────────┴────────┴─────────────────┘

        ```
    """
    return pl.DataFrame([r.to_flat_dict() if unnest_columns else r.to_dict() for r in examples])
