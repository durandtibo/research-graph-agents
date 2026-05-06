from __future__ import annotations

from argos.meta_agent.predictions import BasePrediction, Prediction

################################
#     Tests for Prediction     #
################################


def test_prediction_is_instance_of_base_prediction() -> None:
    assert isinstance(Prediction(example_id="q1", prediction="4"), BasePrediction)


def test_prediction_example_id() -> None:
    assert Prediction(example_id="q1", prediction="4").example_id == "q1"


def test_prediction_prediction() -> None:
    assert Prediction(example_id="q1", prediction="4").prediction == "4"


def test_prediction_metadata_default() -> None:
    assert Prediction(example_id="q1", prediction="4").metadata is None


def test_prediction_metadata() -> None:
    assert Prediction(example_id="q1", prediction="4", metadata={"source": "math"}).metadata == {
        "source": "math"
    }


def test_prediction_equal_true() -> None:
    assert Prediction(example_id="q1", prediction="4").equal(
        Prediction(example_id="q1", prediction="4")
    )


def test_prediction_equal_true_with_metadata() -> None:
    assert Prediction(example_id="q1", prediction="4", metadata={"source": "math"}).equal(
        Prediction(example_id="q1", prediction="4", metadata={"source": "math"})
    )


def test_prediction_equal_false_different_example_id() -> None:
    assert not Prediction(example_id="q1", prediction="4").equal(
        Prediction(example_id="q2", prediction="4")
    )


def test_prediction_equal_false_different_prediction() -> None:
    assert not Prediction(example_id="q1", prediction="4").equal(
        Prediction(example_id="q1", prediction="5")
    )


def test_prediction_equal_false_different_metadata() -> None:
    assert not Prediction(example_id="q1", prediction="4", metadata={"source": "math"}).equal(
        Prediction(example_id="q1", prediction="4", metadata={"source": "science"})
    )


def test_prediction_equal_false_metadata_vs_none() -> None:
    assert not Prediction(example_id="q1", prediction="4", metadata={"source": "math"}).equal(
        Prediction(example_id="q1", prediction="4")
    )


def test_prediction_equal_false_different_type() -> None:
    assert not Prediction(example_id="q1", prediction="4").equal(
        {"example_id": "q1", "input": "What is 2+2?", "prediction": "4", "metadata": None}
    )


def test_prediction_equal_false_different_type_child() -> None:
    class ChildPrediction(Prediction): ...

    assert not Prediction(example_id="q1", prediction="4").equal(
        ChildPrediction(example_id="q1", prediction="4")
    )


def test_prediction_equal_nan_false_by_default() -> None:
    assert not Prediction(example_id="q1", prediction=float("nan")).equal(
        Prediction(example_id="q1", prediction=float("nan"))
    )


def test_prediction_equal_nan_true() -> None:
    assert Prediction(example_id="q1", prediction=float("nan")).equal(
        Prediction(example_id="q1", prediction=float("nan")), equal_nan=True
    )


def test_prediction_to_dict() -> None:
    prediction = Prediction(example_id="q1", prediction="4")
    assert prediction.to_dict() == {"example_id": "q1", "prediction": "4", "metadata": None}


def test_prediction_to_dict_with_metadata() -> None:
    prediction = Prediction(example_id="q1", prediction="4", metadata={"source": "math"})
    assert prediction.to_dict() == {
        "example_id": "q1",
        "prediction": "4",
        "metadata": {"source": "math"},
    }


def test_prediction_from_dict() -> None:
    prediction = Prediction.from_dict({"example_id": "q1", "prediction": "4"})
    assert prediction == Prediction(example_id="q1", prediction="4")


def test_prediction_from_dict_with_metadata() -> None:
    prediction = Prediction.from_dict(
        {
            "example_id": "q1",
            "input": "What is 2+2?",
            "prediction": "4",
            "metadata": {"source": "math"},
        }
    )
    assert prediction == Prediction(example_id="q1", prediction="4", metadata={"source": "math"})


def test_prediction_from_dict_metadata_defaults_to_none() -> None:
    prediction = Prediction.from_dict(
        {"example_id": "q1", "input": "What is 2+2?", "prediction": "4"}
    )
    assert prediction.metadata is None


def test_prediction_roundtrip() -> None:
    prediction = Prediction(example_id="q1", prediction="4", metadata={"source": "math"})
    assert Prediction.from_dict(prediction.to_dict()) == prediction


def test_prediction_roundtrip_without_metadata() -> None:
    prediction = Prediction(example_id="q1", prediction="4")
    assert Prediction.from_dict(prediction.to_dict()) == prediction


def test_prediction_equality() -> None:
    ex1 = Prediction(example_id="q1", prediction="4")
    ex2 = Prediction(example_id="q1", prediction="4")
    assert ex1 == ex2


def test_prediction_inequality_different_example_id() -> None:
    ex1 = Prediction(example_id="q1", prediction="4")
    ex2 = Prediction(example_id="q2", prediction="4")
    assert ex1 != ex2


def test_prediction_inequality_different_prediction() -> None:
    ex1 = Prediction(example_id="q1", prediction="4")
    ex2 = Prediction(example_id="q1", prediction="5")
    assert ex1 != ex2


def test_prediction_repr() -> None:
    prediction = Prediction(example_id="q1", prediction="4")
    assert repr(prediction) == "Prediction(example_id='q1', prediction='4', metadata=None)"


def test_prediction_repr_with_metadata() -> None:
    prediction = Prediction(example_id="q1", prediction="4", metadata={"source": "math"})
    assert repr(prediction) == (
        "Prediction(example_id='q1', prediction='4', metadata={'source': 'math'})"
    )


def test_prediction_str() -> None:
    prediction = Prediction(example_id="q1", prediction="4")
    assert str(prediction) == "Prediction(example_id='q1', prediction='4', metadata=None)"


def test_prediction_str_with_metadata() -> None:
    prediction = Prediction(example_id="q1", prediction="4", metadata={"source": "math"})
    assert str(prediction) == (
        "Prediction(example_id='q1', prediction='4', metadata={'source': 'math'})"
    )
