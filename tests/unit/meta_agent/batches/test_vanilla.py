from __future__ import annotations

from dataclasses import FrozenInstanceError

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from argos.meta_agent.batches import Batch
from argos.meta_agent.entities import LabeledExample

###########################
#     Tests for Batch     #
###########################


def test_batch_items() -> None:
    items = {
        "q1": LabeledExample(id="q1", input="What is 2+2?", target="4"),
        "q2": LabeledExample(id="q2", input="What is 3+3?", target="6"),
    }
    assert Batch(items=items).items == items


def test_batch_metadata_default() -> None:
    assert Batch(items={}).metadata is None


def test_batch_metadata() -> None:
    assert Batch(items={}, metadata={"source": "math"}).metadata == {"source": "math"}


def test_batch_is_frozen() -> None:
    batch = Batch(items={})
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'items'"):
        batch.items = {}


def test_batch_equal_true() -> None:
    items = {"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")}
    assert Batch(items=items).equal(Batch(items=items))


def test_batch_equal_true_empty() -> None:
    assert Batch(items={}).equal(Batch(items={}))


def test_batch_equal_true_with_metadata() -> None:
    items = {"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")}
    assert Batch(items=items, metadata={"source": "math"}).equal(
        Batch(items=items, metadata={"source": "math"})
    )


def test_batch_equal_false_different_items() -> None:
    assert not Batch(items={"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")}).equal(
        Batch(items={"q1": LabeledExample(id="q1", input="What is 2+2?", target="5")})
    )


def test_batch_equal_false_different_keys() -> None:
    item = LabeledExample(id="q1", input="What is 2+2?", target="4")
    assert not Batch(items={"q1": item}).equal(Batch(items={"q2": item}))


def test_batch_equal_false_different_number_of_items() -> None:
    assert not Batch(items={"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")}).equal(
        Batch(
            items={
                "q1": LabeledExample(id="q1", input="What is 2+2?", target="4"),
                "q2": LabeledExample(id="q2", input="What is 3+3?", target="6"),
            }
        )
    )


def test_batch_equal_false_different_metadata() -> None:
    items = {"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")}
    assert not Batch(items=items, metadata={"source": "math"}).equal(
        Batch(items=items, metadata={"source": "science"})
    )


def test_batch_equal_false_metadata_vs_none() -> None:
    items = {"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")}
    assert not Batch(items=items, metadata={"source": "math"}).equal(Batch(items=items))


def test_batch_equal_false_different_type() -> None:
    assert not Batch(items={}).equal({})


def test_batch_equal_false_different_type_child() -> None:
    class Child(Batch): ...

    assert not Batch(items={}).equal(Child({}))


def test_batch_equal_nan_false_by_default() -> None:
    assert not Batch(items={"q1": LabeledExample(id="q1", input=float("nan"), target="4")}).equal(
        Batch(items={"q1": LabeledExample(id="q1", input=float("nan"), target="4")})
    )


def test_batch_equal_nan_true() -> None:
    assert Batch(items={"q1": LabeledExample(id="q1", input=float("nan"), target="4")}).equal(
        Batch(items={"q1": LabeledExample(id="q1", input=float("nan"), target="4")}),
        equal_nan=True,
    )


def test_batch_equal_nan_false_by_default_in_metadata() -> None:
    items = {"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")}
    assert not Batch(items=items, metadata={"score": float("nan")}).equal(
        Batch(items=items, metadata={"score": float("nan")})
    )


def test_batch_equal_nan_true_in_metadata() -> None:
    items = {"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")}
    assert Batch(items=items, metadata={"score": float("nan")}).equal(
        Batch(items=items, metadata={"score": float("nan")}), equal_nan=True
    )


def test_batch_from_list_empty() -> None:
    assert Batch.from_list([]).equal(Batch(items={}))


def test_batch_from_list_single() -> None:
    assert Batch.from_list([LabeledExample(id="q1", input="What is 2+2?", target="4")]).equal(
        Batch(items={"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")})
    )


def test_batch_from_list_multiple() -> None:
    assert Batch.from_list(
        [
            LabeledExample(id="q1", input="What is 2+2?", target="4"),
            LabeledExample(id="q2", input="What is 3+3?", target="6"),
        ]
    ).equal(
        Batch(
            items={
                "q1": LabeledExample(id="q1", input="What is 2+2?", target="4"),
                "q2": LabeledExample(id="q2", input="What is 3+3?", target="6"),
            }
        )
    )


def test_batch_from_list_with_metadata() -> None:
    assert Batch.from_list(
        [LabeledExample(id="q1", input="What is 2+2?", target="4")],
        metadata={"source": "math"},
    ).equal(
        Batch(
            items={"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")},
            metadata={"source": "math"},
        )
    )


def test_batch_from_list_duplicated_ids() -> None:
    with pytest.raises(ValueError, match="duplicated"):
        Batch.from_list(
            [
                LabeledExample(id="q1", input="What is 2+2?", target="4"),
                LabeledExample(id="q1", input="What is 3+3?", target="6"),
            ]
        )


def test_batch_to_dataframe_empty() -> None:
    assert_frame_equal(Batch(items={}).to_dataframe(), pl.DataFrame({}))


def test_batch_to_dataframe_single() -> None:
    assert_frame_equal(
        Batch(
            items={"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")}
        ).to_dataframe(),
        pl.DataFrame(
            {"id": ["q1"], "input": ["What is 2+2?"], "target": ["4"], "metadata": [None]},
            schema={"id": pl.String, "input": pl.String, "target": pl.String, "metadata": pl.Null},
        ),
    )


def test_batch_to_dataframe_multiple() -> None:
    assert_frame_equal(
        Batch(
            items={
                "q1": LabeledExample(id="q1", input="What is 2+2?", target="4"),
                "q2": LabeledExample(id="q2", input="What is 3+3?", target="6"),
            }
        ).to_dataframe(),
        pl.DataFrame(
            {
                "id": ["q1", "q2"],
                "input": ["What is 2+2?", "What is 3+3?"],
                "target": ["4", "6"],
                "metadata": [None, None],
            },
            schema={"id": pl.String, "input": pl.String, "target": pl.String, "metadata": pl.Null},
        ),
    )


def test_batch_to_dataframe_excludes_batch_metadata() -> None:
    batch = Batch(
        items={"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")},
        metadata={"source": "math"},
    )
    assert "source" not in batch.to_dataframe().columns


def test_batch_from_dataframe() -> None:
    frame = pl.DataFrame(
        {"id": ["q1"], "input": ["What is 2+2?"], "target": ["4"], "metadata": [None]},
        schema={"id": pl.String, "input": pl.String, "target": pl.String, "metadata": pl.Null},
    )
    assert Batch.from_dataframe(frame, entity_type=LabeledExample).equal(
        Batch(items={"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")})
    )


def test_batch_from_dataframe_with_metadata() -> None:
    frame = pl.DataFrame(
        {"id": ["q1"], "input": ["What is 2+2?"], "target": ["4"], "metadata": [None]},
        schema={"id": pl.String, "input": pl.String, "target": pl.String, "metadata": pl.Null},
    )
    assert Batch.from_dataframe(
        frame, metadata={"source": "math"}, entity_type=LabeledExample
    ).equal(
        Batch(
            items={"q1": LabeledExample(id="q1", input="What is 2+2?", target="4")},
            metadata={"source": "math"},
        )
    )


def test_batch_roundtrip() -> None:
    original = Batch.from_list(
        [
            LabeledExample(id="q1", input="What is 2+2?", target="4"),
            LabeledExample(id="q2", input="What is 3+3?", target="6"),
        ]
    )
    assert Batch.from_dataframe(original.to_dataframe(), entity_type=LabeledExample).equal(original)
