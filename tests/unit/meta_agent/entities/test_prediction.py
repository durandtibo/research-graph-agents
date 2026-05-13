from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from argos.meta_agent.entities import Prediction

################################
#     Tests for Prediction     #
################################


def test_prediction_id() -> None:
    assert Prediction(id="q1", prediction="5").id == "q1"


def test_prediction_prediction() -> None:
    assert Prediction(id="q1", prediction="5").prediction == "5"


def test_prediction_metadata_default() -> None:
    assert Prediction(id="q1", prediction="5").metadata is None


def test_prediction_metadata() -> None:
    assert Prediction(id="q1", prediction="5", metadata={"source": "math"}).metadata == {
        "source": "math"
    }


def test_prediction_is_frozen() -> None:
    prediction = Prediction(id="q1", prediction="5")
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'id'"):
        prediction.id = "q2"


def test_prediction_equal_true() -> None:
    assert Prediction(id="q1", prediction="5").equal(Prediction(id="q1", prediction="5"))


def test_prediction_equal_true_with_metadata() -> None:
    assert Prediction(id="q1", prediction="5", metadata={"source": "math"}).equal(
        Prediction(id="q1", prediction="5", metadata={"source": "math"})
    )


def test_prediction_equal_false_different_id() -> None:
    assert not Prediction(id="q1", prediction="5").equal(Prediction(id="q2", prediction="5"))


def test_prediction_equal_false_different_prediction() -> None:
    assert not Prediction(id="q1", prediction="5").equal(Prediction(id="q1", prediction="6"))


def test_prediction_equal_false_different_metadata() -> None:
    assert not Prediction(id="q1", prediction="5", metadata={"source": "math"}).equal(
        Prediction(id="q1", prediction="5", metadata={"source": "science"})
    )


def test_prediction_equal_false_metadata_vs_none() -> None:
    assert not Prediction(id="q1", prediction="5", metadata={"source": "math"}).equal(
        Prediction(id="q1", prediction="5")
    )


def test_prediction_equal_false_different_type() -> None:
    assert not Prediction(id="q1", prediction="5").equal(
        {"id": "q1", "prediction": "5", "metadata": None}
    )


def test_prediction_equal_false_different_type_child() -> None:
    class Child(Prediction): ...

    assert not Prediction(id="q1", prediction="5").equal(Child(id="q1", prediction="5"))


def test_prediction_equal_nan_false_by_default() -> None:
    assert not Prediction(id="q1", prediction=float("nan")).equal(
        Prediction(id="q1", prediction=float("nan"))
    )


def test_prediction_equal_nan_true() -> None:
    assert Prediction(id="q1", prediction=float("nan")).equal(
        Prediction(id="q1", prediction=float("nan")), equal_nan=True
    )


def test_prediction_from_dict() -> None:
    assert Prediction.from_dict({"id": "q1", "prediction": "5"}) == Prediction(
        id="q1", prediction="5"
    )


def test_prediction_from_dict_with_metadata() -> None:
    assert Prediction.from_dict(
        {"id": "q1", "prediction": "5", "metadata": {"source": "math"}}
    ) == (Prediction(id="q1", prediction="5", metadata={"source": "math"}))


def test_prediction_from_dict_metadata_defaults_to_none() -> None:
    assert Prediction.from_dict({"id": "q1", "prediction": "5"}).metadata is None


def test_prediction_from_dict_missing_id() -> None:
    with pytest.raises(KeyError):
        Prediction.from_dict({"prediction": "5"})


def test_prediction_from_dict_missing_prediction() -> None:
    with pytest.raises(KeyError):
        Prediction.from_dict({"id": "q1"})


def test_prediction_to_dict() -> None:
    assert Prediction(id="q1", prediction="5").to_dict() == {
        "id": "q1",
        "prediction": "5",
        "metadata": None,
    }


def test_prediction_to_dict_with_metadata() -> None:
    assert Prediction(id="q1", prediction="5", metadata={"source": "math"}).to_dict() == {
        "id": "q1",
        "prediction": "5",
        "metadata": {"source": "math"},
    }


def test_prediction_to_flat_dict() -> None:
    assert Prediction(id="q1", prediction="5").to_flat_dict() == {
        "id": "q1",
        "prediction": "5",
        "metadata": None,
    }


def test_prediction_to_flat_dict_with_metadata() -> None:
    assert Prediction(id="q1", prediction="5", metadata={"source": "math"}).to_flat_dict() == {
        "id": "q1",
        "prediction": "5",
        "metadata.source": "math",
    }


def test_prediction_to_flat_dict_nested_prediction() -> None:
    assert Prediction(id="q1", prediction={"answer": 4, "style": "math"}).to_flat_dict() == {
        "id": "q1",
        "prediction.answer": 4,
        "prediction.style": "math",
        "metadata": None,
    }


def test_prediction_to_flat_dict_custom_separator() -> None:
    assert Prediction(id="q1", prediction="5", metadata={"source": "math"}).to_flat_dict(
        separator="/"
    ) == {
        "id": "q1",
        "prediction": "5",
        "metadata/source": "math",
    }


def test_prediction_to_flat_dict_deeply_nested() -> None:
    assert Prediction(id="q1", prediction={"a": {"b": "c"}}).to_flat_dict() == {
        "id": "q1",
        "prediction.a.b": "c",
        "metadata": None,
    }


def test_prediction_roundtrip() -> None:
    example = Prediction(id="q1", prediction="5", metadata={"source": "math"})
    assert Prediction.from_dict(example.to_dict()) == example


def test_prediction_roundtrip_without_metadata() -> None:
    example = Prediction(id="q1", prediction="5")
    assert Prediction.from_dict(example.to_dict()) == example
