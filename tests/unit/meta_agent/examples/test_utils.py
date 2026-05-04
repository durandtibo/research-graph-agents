from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from argos.meta_agent.examples import Example, examples_to_dataframe

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
