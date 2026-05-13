from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from argos.meta_agent.entities import Record

############################
#     Tests for Record     #
############################


def test_record_id() -> None:
    assert Record(id="q1", input="What is 2+2?", target="4", prediction="5").id == "q1"


def test_record_input() -> None:
    assert Record(id="q1", input="What is 2+2?", target="4", prediction="5").input == "What is 2+2?"


def test_record_target() -> None:
    assert Record(id="q1", input="What is 2+2?", target="4", prediction="5").target == "4"


def test_record_prediction() -> None:
    assert Record(id="q1", input="What is 2+2?", target="4", prediction="5").prediction == "5"


def test_record_metadata_default() -> None:
    assert Record(id="q1").metadata is None


def test_record_input_default() -> None:
    assert Record(id="q1").input is None


def test_record_target_default() -> None:
    assert Record(id="q1").target is None


def test_record_prediction_default() -> None:
    assert Record(id="q1").prediction is None


def test_record_metadata() -> None:
    assert Record(id="q1", metadata={"source": "math"}).metadata == {"source": "math"}


def test_record_is_frozen() -> None:
    record = Record(id="q1")
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'id'"):
        record.id = "q2"


def test_record_equal_true() -> None:
    assert Record(id="q1", input="What is 2+2?", target="4", prediction="5").equal(
        Record(id="q1", input="What is 2+2?", target="4", prediction="5")
    )


def test_record_equal_true_with_metadata() -> None:
    assert Record(
        id="q1", input="What is 2+2?", target="4", prediction="5", metadata={"source": "math"}
    ).equal(
        Record(
            id="q1", input="What is 2+2?", target="4", prediction="5", metadata={"source": "math"}
        )
    )


def test_record_equal_false_different_id() -> None:
    assert not Record(id="q1", input="What is 2+2?", target="4", prediction="5").equal(
        Record(id="q2", input="What is 2+2?", target="4", prediction="5")
    )


def test_record_equal_false_different_input() -> None:
    assert not Record(id="q1", input="What is 2+2?", target="4", prediction="5").equal(
        Record(id="q1", input="What is 4+2?", target="4", prediction="5")
    )


def test_record_equal_false_different_target() -> None:
    assert not Record(id="q1", input="What is 2+2?", target="4", prediction="5").equal(
        Record(id="q1", input="What is 2+2?", target="5", prediction="5")
    )


def test_record_equal_false_different_prediction() -> None:
    assert not Record(id="q1", input="What is 2+2?", target="4", prediction="5").equal(
        Record(id="q1", input="What is 2+2?", target="4", prediction="6")
    )


def test_record_equal_false_different_metadata() -> None:
    assert not Record(id="q1", metadata={"source": "math"}).equal(
        Record(id="q1", metadata={"source": "science"})
    )


def test_record_equal_false_metadata_vs_none() -> None:
    assert not Record(id="q1", metadata={"source": "math"}).equal(Record(id="q1"))


def test_record_equal_false_different_type() -> None:
    assert not Record(id="q1").equal(
        {"id": "q1", "input": None, "target": None, "prediction": None, "metadata": None}
    )


def test_record_equal_false_different_type_child() -> None:
    class Child(Record): ...

    assert not Record(id="q1").equal(Child(id="q1"))


def test_record_equal_nan_false_by_default() -> None:
    assert not Record(id="q1", input=float("nan")).equal(Record(id="q1", input=float("nan")))


def test_record_equal_nan_true() -> None:
    assert Record(id="q1", input=float("nan")).equal(
        Record(id="q1", input=float("nan")), equal_nan=True
    )


def test_record_from_dict() -> None:
    assert Record.from_dict(
        {"id": "q1", "input": "What is 2+2?", "target": "4", "prediction": "5"}
    ) == (Record(id="q1", input="What is 2+2?", target="4", prediction="5"))


def test_record_from_dict_with_metadata() -> None:
    assert Record.from_dict(
        {
            "id": "q1",
            "input": "What is 2+2?",
            "target": "4",
            "prediction": "5",
            "metadata": {"source": "math"},
        }
    ) == Record(
        id="q1", input="What is 2+2?", target="4", prediction="5", metadata={"source": "math"}
    )


def test_record_from_dict_missing_input_defaults_to_none() -> None:
    assert Record.from_dict({"id": "q1"}).input is None


def test_record_from_dict_missing_target_defaults_to_none() -> None:
    assert Record.from_dict({"id": "q1"}).target is None


def test_record_from_dict_missing_prediction_defaults_to_none() -> None:
    assert Record.from_dict({"id": "q1"}).prediction is None


def test_record_from_dict_metadata_defaults_to_none() -> None:
    assert Record.from_dict({"id": "q1"}).metadata is None


def test_record_from_dict_missing_id() -> None:
    with pytest.raises(KeyError):
        Record.from_dict({"input": "What is 2+2?", "target": "4", "prediction": "5"})


def test_record_to_dict() -> None:
    assert Record(id="q1", input="What is 2+2?", target="4", prediction="5").to_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "prediction": "5",
        "metadata": None,
    }


def test_record_to_dict_with_metadata() -> None:
    assert Record(
        id="q1", input="What is 2+2?", target="4", prediction="5", metadata={"source": "math"}
    ).to_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "prediction": "5",
        "metadata": {"source": "math"},
    }


def test_record_to_dict_defaults() -> None:
    assert Record(id="q1").to_dict() == {
        "id": "q1",
        "input": None,
        "target": None,
        "prediction": None,
        "metadata": None,
    }


def test_record_to_flat_dict() -> None:
    assert Record(id="q1", input="What is 2+2?", target="4", prediction="5").to_flat_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "prediction": "5",
        "metadata": None,
    }


def test_record_to_flat_dict_with_metadata() -> None:
    assert Record(
        id="q1", input="What is 2+2?", target="4", prediction="5", metadata={"source": "math"}
    ).to_flat_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "prediction": "5",
        "metadata.source": "math",
    }


def test_record_to_flat_dict_nested_target() -> None:
    assert Record(
        id="q1", input="What is 2+2?", target={"answer": 4, "style": "math"}
    ).to_flat_dict() == {
        "id": "q1",
        "input": "What is 2+2?",
        "target.answer": 4,
        "target.style": "math",
        "prediction": None,
        "metadata": None,
    }


def test_record_to_flat_dict_nested_prediction() -> None:
    assert Record(id="q1", prediction={"answer": 4, "style": "math"}).to_flat_dict() == {
        "id": "q1",
        "input": None,
        "target": None,
        "prediction.answer": 4,
        "prediction.style": "math",
        "metadata": None,
    }


def test_record_to_flat_dict_custom_separator() -> None:
    assert Record(
        id="q1", input="What is 2+2?", target="4", prediction="5", metadata={"source": "math"}
    ).to_flat_dict(separator="/") == {
        "id": "q1",
        "input": "What is 2+2?",
        "target": "4",
        "prediction": "5",
        "metadata/source": "math",
    }


def test_record_to_flat_dict_deeply_nested() -> None:
    assert Record(id="q1", target={"a": {"b": "c"}}).to_flat_dict() == {
        "id": "q1",
        "input": None,
        "target.a.b": "c",
        "prediction": None,
        "metadata": None,
    }


def test_record_roundtrip() -> None:
    record = Record(
        id="q1", input="What is 2+2?", target="4", prediction="5", metadata={"source": "math"}
    )
    assert Record.from_dict(record.to_dict()) == record


def test_record_roundtrip_defaults() -> None:
    record = Record(id="q1")
    assert Record.from_dict(record.to_dict()) == record
