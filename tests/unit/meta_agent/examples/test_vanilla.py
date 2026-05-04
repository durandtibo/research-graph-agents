from __future__ import annotations

from argos.meta_agent.examples import Example

#############################
#     Tests for Example     #
#############################


def test_example_id() -> None:
    assert Example(id="q1", input="What is 2+2?", target="4").id == "q1"


def test_example_input() -> None:
    assert Example(id="q1", input="What is 2+2?", target="4").input == "What is 2+2?"


def test_example_target() -> None:
    assert Example(id="q1", input="What is 2+2?", target="4").target == "4"


def test_example_metadata_default() -> None:
    assert Example(id="q1", input="What is 2+2?", target="4").metadata is None


def test_example_metadata() -> None:
    assert Example(
        id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}
    ).metadata == {"source": "math"}


def test_example_to_dict() -> None:
    example = Example(id="q1", input="What is 2+2?", target="4")
    assert example.to_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "metadata": None,
    }


def test_example_to_dict_with_metadata() -> None:
    example = Example(id="q1", input="What is 2+2?", target="4", metadata={"source": "math"})
    assert example.to_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "metadata": {"source": "math"},
    }


def test_example_from_dict() -> None:
    example = Example.from_dict({"id": "q1", "input": "What is 2+2?", "target": "4"})
    assert example == Example(id="q1", input="What is 2+2?", target="4")


def test_example_from_dict_with_metadata() -> None:
    example = Example.from_dict(
        {
            "id": "q1",
            "input": "What is 2+2?",
            "target": "4",
            "metadata": {"source": "math"},
        }
    )
    assert example == Example(
        id="q1", input="What is 2+2?", target="4", metadata={"source": "math"}
    )


def test_example_from_dict_metadata_defaults_to_none() -> None:
    example = Example.from_dict({"id": "q1", "input": "What is 2+2?", "target": "4"})
    assert example.metadata is None


def test_example_roundtrip() -> None:
    example = Example(id="q1", input="What is 2+2?", target="4", metadata={"source": "math"})
    assert Example.from_dict(example.to_dict()) == example


def test_example_roundtrip_without_metadata() -> None:
    example = Example(id="q1", input="What is 2+2?", target="4")
    assert Example.from_dict(example.to_dict()) == example
