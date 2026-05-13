from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from argos.meta_agent.entities import (
    Record,
    dataframe_to_entities,
    entities_to_dataframe,
)


class CustomRecord(Record): ...


###########################################
#     Tests for dataframe_to_entities     #
###########################################


def test_dataframe_to_entities_empty() -> None:
    assert dataframe_to_entities(pl.DataFrame({}), entity_type=Record) == []


def test_dataframe_to_entities_single() -> None:
    frame = pl.DataFrame(
        {
            "id": ["q1"],
            "input": ["What is 2+2?"],
            "target": ["4"],
            "prediction": [None],
            "metadata": [None],
        },
        schema={
            "id": pl.String,
            "input": pl.String,
            "target": pl.String,
            "prediction": pl.Null,
            "metadata": pl.Null,
        },
    )
    assert dataframe_to_entities(frame, entity_type=Record) == [
        Record(id="q1", input="What is 2+2?", target="4")
    ]


def test_dataframe_to_entities_multiple() -> None:
    frame = pl.DataFrame(
        {
            "id": ["q1", "q2"],
            "input": ["What is 2+2?", "What is 4+2?"],
            "target": ["4", "6"],
            "prediction": [None, None],
            "metadata": [None, None],
        },
        schema={
            "id": pl.String,
            "input": pl.String,
            "target": pl.String,
            "prediction": pl.Null,
            "metadata": pl.Null,
        },
    )
    assert dataframe_to_entities(frame, entity_type=Record) == [
        Record(id="q1", input="What is 2+2?", target="4"),
        Record(id="q2", input="What is 4+2?", target="6"),
    ]


def test_dataframe_to_entities_with_metadata() -> None:
    frame = pl.DataFrame(
        {
            "id": ["q1"],
            "input": ["What is 2+2?"],
            "target": ["4"],
            "prediction": [None],
            "metadata": [{"source": "math"}],
        },
        schema={
            "id": pl.String,
            "input": pl.String,
            "target": pl.String,
            "prediction": pl.Null,
            "metadata": pl.Struct({"source": pl.String}),
        },
    )
    assert dataframe_to_entities(frame, entity_type=Record) == [
        Record(id="q1", input="What is 2+2?", target="4", metadata={"source": "math"})
    ]


def test_dataframe_to_entities_preserves_order() -> None:
    frame = pl.DataFrame(
        {
            "id": ["q3", "q1", "q2"],
            "input": ["c", "a", "b"],
            "target": ["3", "1", "2"],
            "prediction": [None, None, None],
            "metadata": [None, None, None],
        },
        schema={
            "id": pl.String,
            "input": pl.String,
            "target": pl.String,
            "prediction": pl.Null,
            "metadata": pl.Null,
        },
    )
    assert dataframe_to_entities(frame, entity_type=Record) == [
        Record(id="q3", input="c", target="3"),
        Record(id="q1", input="a", target="1"),
        Record(id="q2", input="b", target="2"),
    ]


def test_dataframe_to_entities_custom_type() -> None:
    frame = pl.DataFrame(
        {
            "id": ["q1", "q2"],
            "input": ["What is 2+2?", "What is 4+2?"],
            "target": ["4", "6"],
            "prediction": [None, None],
            "metadata": [None, None],
        },
        schema={
            "id": pl.String,
            "input": pl.String,
            "target": pl.String,
            "prediction": pl.Null,
            "metadata": pl.Null,
        },
    )
    entities = dataframe_to_entities(frame, entity_type=CustomRecord)
    assert entities == [
        CustomRecord(id="q1", input="What is 2+2?", target="4"),
        CustomRecord(id="q2", input="What is 4+2?", target="6"),
    ]
    assert entities != [
        Record(id="q1", input="What is 2+2?", target="4"),
        Record(id="q2", input="What is 4+2?", target="6"),
    ]


def test_dataframe_to_entities_roundtrip() -> None:
    original = [
        Record(id="q1", input="What is 2+2?", target="4"),
        Record(id="q2", input="What is 4+2?", target="6"),
    ]
    assert dataframe_to_entities(entities_to_dataframe(original), entity_type=Record) == original


###########################################
#     Tests for entities_to_dataframe     #
###########################################


def test_entities_to_dataframe_empty() -> None:
    assert_frame_equal(entities_to_dataframe([]), pl.DataFrame({}))


def test_entities_to_dataframe_single() -> None:
    assert_frame_equal(
        entities_to_dataframe([Record(id="q1", input="What is 2+2?", target="4")]),
        pl.DataFrame(
            {
                "id": ["q1"],
                "input": ["What is 2+2?"],
                "target": ["4"],
                "prediction": [None],
                "metadata": [None],
            },
            schema={
                "id": pl.String,
                "input": pl.String,
                "target": pl.String,
                "prediction": pl.Null,
                "metadata": pl.Null,
            },
        ),
    )


def test_entities_to_dataframe_multiple() -> None:
    assert_frame_equal(
        entities_to_dataframe(
            [
                Record(id="q1", input="What is 2+2?", target="4"),
                Record(id="q2", input="What is 4+2?", target="6"),
            ]
        ),
        pl.DataFrame(
            {
                "id": ["q1", "q2"],
                "input": ["What is 2+2?", "What is 4+2?"],
                "target": ["4", "6"],
                "prediction": [None, None],
                "metadata": [None, None],
            },
            schema={
                "id": pl.String,
                "input": pl.String,
                "target": pl.String,
                "prediction": pl.Null,
                "metadata": pl.Null,
            },
        ),
    )


def test_entities_to_dataframe_with_metadata() -> None:
    assert_frame_equal(
        entities_to_dataframe(
            [Record(id="q1", input="What is 2+2?", target="4", metadata={"source": "math"})]
        ),
        pl.DataFrame(
            {
                "id": ["q1"],
                "input": ["What is 2+2?"],
                "target": ["4"],
                "prediction": [None],
                "metadata": [{"source": "math"}],
            },
            schema={
                "id": pl.String,
                "input": pl.String,
                "target": pl.String,
                "prediction": pl.Null,
                "metadata": pl.Struct({"source": pl.String}),
            },
        ),
    )


def test_entities_to_dataframe_unnest_columns_false_by_default() -> None:
    assert_frame_equal(
        entities_to_dataframe(
            [
                Record(id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}),
                Record(id="q2", input="What is 4+2?", target="6", metadata={"source": "science"}),
            ]
        ),
        pl.DataFrame(
            {
                "id": ["q1", "q2"],
                "input": ["What is 2+2?", "What is 4+2?"],
                "target": ["4", "6"],
                "prediction": [None, None],
                "metadata": [{"source": "math"}, {"source": "science"}],
            },
            schema={
                "id": pl.String,
                "input": pl.String,
                "target": pl.String,
                "prediction": pl.Null,
                "metadata": pl.Struct({"source": pl.String}),
            },
        ),
    )


def test_entities_to_dataframe_unnest_columns_true() -> None:
    assert_frame_equal(
        entities_to_dataframe(
            [
                Record(id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}),
                Record(id="q2", input="What is 4+2?", target="6", metadata={"source": "science"}),
            ],
            unnest_columns=True,
        ),
        pl.DataFrame(
            {
                "id": ["q1", "q2"],
                "input": ["What is 2+2?", "What is 4+2?"],
                "target": ["4", "6"],
                "prediction": [None, None],
                "metadata.source": ["math", "science"],
            },
            schema={
                "id": pl.String,
                "input": pl.String,
                "target": pl.String,
                "prediction": pl.Null,
                "metadata.source": pl.String,
            },
        ),
    )


def test_entities_to_dataframe_roundtrip() -> None:
    original = [
        Record(id="q1", input="What is 2+2?", target="4"),
        Record(id="q2", input="What is 4+2?", target="6"),
    ]
    assert dataframe_to_entities(entities_to_dataframe(original), entity_type=Record) == original
