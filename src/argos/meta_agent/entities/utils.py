r"""Contain utilities for entities."""

from __future__ import annotations

__all__ = ["dataframe_to_entities", "entities_to_dataframe"]

from typing import TypeVar

import polars as pl

from argos.meta_agent.entities import BaseEntity

EntityT = TypeVar("EntityT", bound=BaseEntity)


def dataframe_to_entities(frame: pl.DataFrame, entity_type: type[EntityT]) -> list[EntityT]:
    r"""Convert a Polars DataFrame into a list of entities.

    Each row in the DataFrame is converted into a single entity using
    the ``from_dict`` class method of ``entity_type``. This is the
    inverse of :func:`entities_to_dataframe`.

    Args:
        frame: A Polars DataFrame where each row represents a single
            entity. Column names must match the fields expected by
            ``entity_type.from_dict``.
        entity_type: The entity class to instantiate for each row.
            Defaults to :class:`Example`.

    Returns:
        A list of entities, one per row, in the same order as the
            DataFrame.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.meta_agent.entities import Record, dataframe_to_entities
        >>> frame = pl.DataFrame(
        ...     {
        ...         "id": ["q1", "q2"],
        ...         "input": ["What is 2+2?", "What is 4+2?"],
        ...         "target": ["4", "6"],
        ...         "metadata": [None, None],
        ...     }
        ... )
        >>> entities = dataframe_to_entities(frame, entity_type=Record)
        >>> entities[0]
        Record(id='q1', input='What is 2+2?', target='4', prediction=None, metadata=None)

        ```
    """
    return [entity_type.from_dict(row) for row in frame.iter_rows(named=True)]


def entities_to_dataframe(
    entities: list[BaseEntity],
    *,
    unnest_columns: bool = False,
) -> pl.DataFrame:
    r"""Convert a list of entities into a Polars DataFrame.

    Each entity is represented as a single row, where columns correspond
    to the fields returned by ``to_dict``. This is the recommended way to
    build a DataFrame from multiple entities, as a single entity does not
    carry enough structure to justify a DataFrame on its own.

    Args:
        entities: A list of entities to convert. All entities should have
            consistent keys in their ``to_dict`` output to ensure a
            well-formed DataFrame. An empty list returns an empty DataFrame.
        unnest_columns: If ``True``, nested fields are flattened into
            separate top-level columns using ``to_flat_dict`` (e.g.
            ``metadata.score`` becomes its own column). If ``False``
            (default), nested fields are kept as struct columns.

    Returns:
        A Polars DataFrame with one row per entity and one column per
            field.

    Example:
        ```pycon
        >>> from argos.meta_agent.entities import Record, entities_to_dataframe
        >>> entities = [
        ...     Record(id="q1", input="What is 2+2?", target="4", metadata={"source": "cat"}),
        ...     Record(id="q2", input="What is 4+2?", target="6", metadata={"source": "bear"}),
        ... ]
        >>> frame = entities_to_dataframe(entities)
        >>> frame
        shape: (2, 5)
        ┌─────┬──────────────┬────────┬────────────┬───────────┐
        │ id  ┆ input        ┆ target ┆ prediction ┆ metadata  │
        │ --- ┆ ---          ┆ ---    ┆ ---        ┆ ---       │
        │ str ┆ str          ┆ str    ┆ null       ┆ struct[1] │
        ╞═════╪══════════════╪════════╪════════════╪═══════════╡
        │ q1  ┆ What is 2+2? ┆ 4      ┆ null       ┆ {"cat"}   │
        │ q2  ┆ What is 4+2? ┆ 6      ┆ null       ┆ {"bear"}  │
        └─────┴──────────────┴────────┴────────────┴───────────┘

        >>> frame = entities_to_dataframe(entities, unnest_columns=True)
        >>> frame
        shape: (2, 5)
        ┌─────┬──────────────┬────────┬────────────┬─────────────────┐
        │ id  ┆ input        ┆ target ┆ prediction ┆ metadata.source │
        │ --- ┆ ---          ┆ ---    ┆ ---        ┆ ---             │
        │ str ┆ str          ┆ str    ┆ null       ┆ str             │
        ╞═════╪══════════════╪════════╪════════════╪═════════════════╡
        │ q1  ┆ What is 2+2? ┆ 4      ┆ null       ┆ cat             │
        │ q2  ┆ What is 4+2? ┆ 6      ┆ null       ┆ bear            │
        └─────┴──────────────┴────────┴────────────┴─────────────────┘

        ```
    """
    return pl.DataFrame([r.to_flat_dict() if unnest_columns else r.to_dict() for r in entities])
