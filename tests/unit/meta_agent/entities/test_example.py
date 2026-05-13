from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from argos.meta_agent.entities import Example

#############################
#     Tests for Example     #
#############################


def test_example_id() -> None:
    assert Example(id="q1", input="What is 2+2?").id == "q1"


def test_example_input() -> None:
    assert Example(id="q1", input="What is 2+2?").input == "What is 2+2?"


def test_example_metadata_default() -> None:
    assert Example(id="q1", input="What is 2+2?").metadata is None


def test_example_metadata() -> None:
    assert Example(id="q1", input="What is 2+2?", metadata={"source": "math"}).metadata == {
        "source": "math"
    }


def test_example_is_frozen() -> None:
    example = Example(id="q1", input="What is 2+2?")
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'id'"):
        example.id = "q2"


def test_example_equal_true() -> None:
    assert Example(id="q1", input="What is 2+2?").equal(Example(id="q1", input="What is 2+2?"))


def test_example_equal_true_with_metadata() -> None:
    assert Example(id="q1", input="What is 2+2?", metadata={"source": "math"}).equal(
        Example(id="q1", input="What is 2+2?", metadata={"source": "math"})
    )


def test_example_equal_false_different_id() -> None:
    assert not Example(id="q1", input="What is 2+2?").equal(Example(id="q2", input="What is 2+2?"))


def test_example_equal_false_different_input() -> None:
    assert not Example(id="q1", input="What is 2+2?").equal(Example(id="q1", input="What is 4+2?"))


def test_example_equal_false_different_metadata() -> None:
    assert not Example(id="q1", input="What is 2+2?", metadata={"source": "math"}).equal(
        Example(id="q1", input="What is 2+2?", metadata={"source": "science"})
    )


def test_example_equal_false_metadata_vs_none() -> None:
    assert not Example(id="q1", input="What is 2+2?", metadata={"source": "math"}).equal(
        Example(id="q1", input="What is 2+2?")
    )


def test_example_equal_false_different_type() -> None:
    assert not Example(id="q1", input="What is 2+2?").equal(
        {"id": "q1", "input": "What is 2+2?", "metadata": None}
    )


def test_example_equal_false_different_type_child() -> None:
    class ChildExample(Example): ...

    assert not Example(id="q1", input="What is 2+2?").equal(
        ChildExample(id="q1", input="What is 2+2?")
    )


def test_example_equal_nan_false_by_default() -> None:
    assert not Example(id="q1", input=float("nan")).equal(Example(id="q1", input=float("nan")))


def test_example_equal_nan_true() -> None:
    assert Example(id="q1", input=float("nan")).equal(
        Example(id="q1", input=float("nan")), equal_nan=True
    )


def test_example_from_dict() -> None:
    assert Example.from_dict({"id": "q1", "input": "What is 2+2?"}) == Example(
        id="q1", input="What is 2+2?"
    )


def test_example_from_dict_with_metadata() -> None:
    assert Example.from_dict(
        {"id": "q1", "input": "What is 2+2?", "metadata": {"source": "math"}}
    ) == (Example(id="q1", input="What is 2+2?", metadata={"source": "math"}))


def test_example_from_dict_metadata_defaults_to_none() -> None:
    assert Example.from_dict({"id": "q1", "input": "What is 2+2?"}).metadata is None


def test_example_from_dict_missing_id() -> None:
    with pytest.raises(KeyError):
        Example.from_dict({"input": "What is 2+2?"})


def test_example_from_dict_missing_input() -> None:
    with pytest.raises(KeyError):
        Example.from_dict({"id": "q1"})


def test_example_to_dict() -> None:
    assert Example(id="q1", input="What is 2+2?").to_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "metadata": None,
    }


def test_example_to_dict_with_metadata() -> None:
    assert Example(id="q1", input="What is 2+2?", metadata={"source": "math"}).to_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "metadata": {"source": "math"},
    }


def test_example_roundtrip() -> None:
    example = Example(id="q1", input="What is 2+2?", metadata={"source": "math"})
    assert Example.from_dict(example.to_dict()) == example


def test_example_roundtrip_without_metadata() -> None:
    example = Example(id="q1", input="What is 2+2?")
    assert Example.from_dict(example.to_dict()) == example


def test_example_to_flat_dict() -> None:
    assert Example(id="q1", input="What is 2+2?").to_flat_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "metadata": None,
    }


def test_example_to_flat_dict_with_metadata() -> None:
    assert Example(id="q1", input="What is 2+2?", metadata={"source": "math"}).to_flat_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "metadata.source": "math",
    }


def test_example_to_flat_dict_nested_input() -> None:
    assert Example(
        id="q1", input={"question": "What is 2+2?", "difficulty": "easy"}
    ).to_flat_dict() == {
        "id": "q1",
        "input.question": "What is 2+2?",
        "input.difficulty": "easy",
        "metadata": None,
    }


def test_example_to_flat_dict_custom_separator() -> None:
    assert Example(id="q1", input="What is 2+2?", metadata={"source": "math"}).to_flat_dict(
        separator="/"
    ) == {
        "id": "q1",
        "input": "What is 2+2?",
        "metadata/source": "math",
    }


def test_example_to_flat_dict_deeply_nested() -> None:
    assert Example(id="q1", input={"a": {"b": "c"}}).to_flat_dict() == {
        "id": "q1",
        "input.a.b": "c",
        "metadata": None,
    }
