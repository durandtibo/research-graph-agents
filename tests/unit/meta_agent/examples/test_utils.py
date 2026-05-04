from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from argos.meta_agent.examples import (
    Example,
    dataframe_to_examples,
    examples_to_dataframe,
)


class CustomExample(Example): ...


###########################################
#     Tests for dataframe_to_examples     #
###########################################


def test_dataframe_to_examples_empty() -> None:
    assert dataframe_to_examples(pl.DataFrame({})) == []


def test_dataframe_to_examples_single() -> None:
    frame = pl.DataFrame(
        {
            "id": ["q1"],
            "input": ["What is 2+2?"],
            "target": ["4"],
            "metadata": [None],
        }
    )
    assert dataframe_to_examples(frame) == [Example(id="q1", input="What is 2+2?", target="4")]


def test_dataframe_to_examples_multiple() -> None:
    frame = pl.DataFrame(
        {
            "id": ["q1", "q2"],
            "input": ["What is 2+2?", "What is 4+2?"],
            "target": ["4", "6"],
            "metadata": [None, None],
        }
    )
    assert dataframe_to_examples(frame) == [
        Example(id="q1", input="What is 2+2?", target="4"),
        Example(id="q2", input="What is 4+2?", target="6"),
    ]


def test_dataframe_to_examples_with_metadata() -> None:
    frame = pl.DataFrame(
        {
            "id": ["q1"],
            "input": ["What is 2+2?"],
            "target": ["4"],
            "metadata": [{"source": "math"}],
        }
    )
    assert dataframe_to_examples(frame) == [
        Example(id="q1", input="What is 2+2?", target="4", metadata={"source": "math"})
    ]


def test_dataframe_to_examples_preserves_order() -> None:
    frame = pl.DataFrame(
        {
            "id": ["q3", "q1", "q2"],
            "input": ["c", "a", "b"],
            "target": ["3", "1", "2"],
            "metadata": [None, None, None],
        }
    )
    assert dataframe_to_examples(frame) == [
        Example(id="q3", input="c", target="3"),
        Example(id="q1", input="a", target="1"),
        Example(id="q2", input="b", target="2"),
    ]


def test_dataframe_to_examples_custom_class() -> None:

    examples = dataframe_to_examples(
        frame=pl.DataFrame(
            {
                "id": ["q1", "q2"],
                "input": ["What is 2+2?", "What is 4+2?"],
                "target": ["4", "6"],
                "metadata": [None, None],
            }
        ),
        example_cls=CustomExample,
    )
    assert examples == [
        CustomExample(id="q1", input="What is 2+2?", target="4"),
        CustomExample(id="q2", input="What is 4+2?", target="6"),
    ]
    assert examples != [
        Example(id="q1", input="What is 2+2?", target="4"),
        Example(id="q2", input="What is 4+2?", target="6"),
    ]


def test_dataframe_to_examples_roundtrip() -> None:
    original = [
        Example(id="q1", input="What is 2+2?", target="4"),
        Example(id="q2", input="What is 4+2?", target="6"),
    ]
    assert dataframe_to_examples(examples_to_dataframe(original)) == original


###########################################
#     Tests for examples_to_dataframe     #
###########################################


def test_examples_to_dataframe_empty() -> None:
    frame = examples_to_dataframe([])
    assert_frame_equal(frame, pl.DataFrame({}))


def test_examples_to_dataframe_single() -> None:
    frame = examples_to_dataframe([Example(id="q1", input="What is 2+2?", target="4")])
    assert_frame_equal(
        frame,
        pl.DataFrame(
            {
                "id": ["q1"],
                "input": ["What is 2+2?"],
                "target": ["4"],
                "metadata": [None],
            },
            schema={"id": pl.String, "input": pl.String, "target": pl.String, "metadata": pl.Null},
        ),
    )


def test_examples_to_dataframe_multiple() -> None:
    frame = examples_to_dataframe(
        [
            Example(id="q1", input="What is 2+2?", target="4"),
            Example(id="q2", input="What is 4+2?", target="6"),
        ]
    )
    assert_frame_equal(
        frame,
        pl.DataFrame(
            {
                "id": ["q1", "q2"],
                "input": ["What is 2+2?", "What is 4+2?"],
                "target": ["4", "6"],
                "metadata": [None, None],
            },
            schema={"id": pl.String, "input": pl.String, "target": pl.String, "metadata": pl.Null},
        ),
    )


def test_examples_to_dataframe_with_metadata() -> None:
    frame = examples_to_dataframe(
        [Example(id="q1", input="What is 2+2?", target="4", metadata={"source": "math"})]
    )
    assert_frame_equal(
        frame,
        pl.DataFrame(
            {
                "id": ["q1"],
                "input": ["What is 2+2?"],
                "target": ["4"],
                "metadata": [{"source": "math"}],
            },
            schema={
                "id": pl.String,
                "input": pl.String,
                "target": pl.String,
                "metadata": pl.Struct({"source": pl.String}),
            },
        ),
    )
