from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from argos.meta_agent.entities import LabeledExample

####################################
#     Tests for LabeledExample     #
####################################


def test_labeled_example_id() -> None:
    assert LabeledExample(id="q1", input="What is 2+2?", target="4").id == "q1"


def test_labeled_example_input() -> None:
    assert LabeledExample(id="q1", input="What is 2+2?", target="4").input == "What is 2+2?"


def test_labeled_example_target() -> None:
    assert LabeledExample(id="q1", input="What is 2+2?", target="4").target == "4"


def test_labeled_example_metadata_default() -> None:
    assert LabeledExample(id="q1", input="What is 2+2?", target="4").metadata is None


def test_labeled_example_metadata() -> None:
    assert LabeledExample(
        id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}
    ).metadata == {"source": "math"}


def test_labeled_example_is_frozen() -> None:
    example = LabeledExample(id="q1", input="What is 2+2?", target="4")
    with pytest.raises(FrozenInstanceError):
        example.id = "q2"


def test_labeled_example_equal_true() -> None:
    assert LabeledExample(id="q1", input="What is 2+2?", target="4").equal(
        LabeledExample(id="q1", input="What is 2+2?", target="4")
    )


def test_labeled_example_equal_true_with_metadata() -> None:
    assert LabeledExample(
        id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}
    ).equal(LabeledExample(id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}))


def test_labeled_example_equal_false_different_id() -> None:
    assert not LabeledExample(id="q1", input="What is 2+2?", target="4").equal(
        LabeledExample(id="q2", input="What is 2+2?", target="4")
    )


def test_labeled_example_equal_false_different_input() -> None:
    assert not LabeledExample(id="q1", input="What is 2+2?", target="4").equal(
        LabeledExample(id="q1", input="What is 4+2?", target="4")
    )


def test_labeled_example_equal_false_different_target() -> None:
    assert not LabeledExample(id="q1", input="What is 2+2?", target="4").equal(
        LabeledExample(id="q1", input="What is 2+2?", target="5")
    )


def test_labeled_example_equal_false_different_metadata() -> None:
    assert not LabeledExample(
        id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}
    ).equal(
        LabeledExample(id="q1", input="What is 2+2?", target="4", metadata={"source": "science"})
    )


def test_labeled_example_equal_false_metadata_vs_none() -> None:
    assert not LabeledExample(
        id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}
    ).equal(LabeledExample(id="q1", input="What is 2+2?", target="4"))


def test_labeled_example_equal_false_different_type() -> None:
    assert not LabeledExample(id="q1", input="What is 2+2?", target="4").equal(
        {"id": "q1", "input": "What is 2+2?", "target": "4", "metadata": None}
    )


def test_labeled_example_equal_nan_false_by_default() -> None:
    assert not LabeledExample(id="q1", input=float("nan"), target="4").equal(
        LabeledExample(id="q1", input=float("nan"), target="4")
    )


def test_labeled_example_equal_nan_true() -> None:
    assert LabeledExample(id="q1", input=float("nan"), target="4").equal(
        LabeledExample(id="q1", input=float("nan"), target="4"), equal_nan=True
    )


def test_labeled_example_from_dict() -> None:
    assert LabeledExample.from_dict({"id": "q1", "input": "What is 2+2?", "target": "4"}) == (
        LabeledExample(id="q1", input="What is 2+2?", target="4")
    )


def test_labeled_example_from_dict_with_metadata() -> None:
    assert LabeledExample.from_dict(
        {"id": "q1", "input": "What is 2+2?", "target": "4", "metadata": {"source": "math"}}
    ) == LabeledExample(id="q1", input="What is 2+2?", target="4", metadata={"source": "math"})


def test_labeled_example_from_dict_missing_input_defaults_to_none() -> None:
    assert LabeledExample.from_dict({"id": "q1", "target": "4"}).input is None


def test_labeled_example_from_dict_missing_target_defaults_to_none() -> None:
    assert LabeledExample.from_dict({"id": "q1", "input": "What is 2+2?"}).target is None


def test_labeled_example_from_dict_metadata_defaults_to_none() -> None:
    assert (
        LabeledExample.from_dict({"id": "q1", "input": "What is 2+2?", "target": "4"}).metadata
        is None
    )


def test_labeled_example_from_dict_missing_id() -> None:
    with pytest.raises(KeyError):
        LabeledExample.from_dict({"input": "What is 2+2?", "target": "4"})


# to_dict


def test_labeled_example_to_dict() -> None:
    assert LabeledExample(id="q1", input="What is 2+2?", target="4").to_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "metadata": None,
    }


def test_labeled_example_to_dict_with_metadata() -> None:
    assert LabeledExample(
        id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}
    ).to_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "metadata": {"source": "math"},
    }


def test_labeled_example_to_flat_dict() -> None:
    assert LabeledExample(id="q1", input="What is 2+2?", target="4").to_flat_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "metadata": None,
    }


def test_labeled_example_to_flat_dict_with_metadata() -> None:
    assert LabeledExample(
        id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}
    ).to_flat_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "metadata.source": "math",
    }


def test_labeled_example_to_flat_dict_nested_target() -> None:
    assert LabeledExample(
        id="q1", input="What is 2+2?", target={"answer": 4, "style": "math"}
    ).to_flat_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target.answer": 4,
        "target.style": "math",
        "metadata": None,
    }


def test_labeled_example_to_flat_dict_custom_separator() -> None:
    assert LabeledExample(
        id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}
    ).to_flat_dict(separator="/") == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "metadata/source": "math",
    }


def test_labeled_example_to_flat_dict_deeply_nested() -> None:
    assert LabeledExample(
        id="q1", input="What is 2+2?", target={"a": {"b": "c"}}
    ).to_flat_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target.a.b": "c",
        "metadata": None,
    }


def test_labeled_example_roundtrip() -> None:
    example = LabeledExample(id="q1", input="What is 2+2?", target="4", metadata={"source": "math"})
    assert LabeledExample.from_dict(example.to_dict()) == example


def test_labeled_example_roundtrip_without_metadata() -> None:
    example = LabeledExample(id="q1", input="What is 2+2?", target="4")
    assert LabeledExample.from_dict(example.to_dict()) == example
