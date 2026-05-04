r"""Unit tests for PredictionRecord and PredictionResult."""

import pytest

from argos.meta_agent.prediction import PredictionRecord, PredictionResult

#####################################
#     Tests for PredictionRecord     #
#####################################


def test_prediction_record_default_attributes() -> None:
    record = PredictionRecord(example_id="q1", prediction="4")
    assert record == PredictionRecord(example_id="q1", prediction="4", metadata=None)


def test_prediction_record_custom_metadata() -> None:
    record = PredictionRecord(example_id="q1", prediction="4", metadata={"source": "llm"})
    assert record == PredictionRecord(
        example_id="q1", prediction="4", metadata={"source": "llm"}
    )


@pytest.mark.parametrize(
    "prediction",
    [
        pytest.param(42, id="int"),
        pytest.param(3.14, id="float"),
        pytest.param({"answer": "yes"}, id="dict"),
        pytest.param(True, id="bool"),
    ],
)
def test_prediction_record_supports_various_prediction_types(prediction: object) -> None:
    record = PredictionRecord(example_id="q1", prediction=prediction)
    assert record.prediction == prediction


######################################
#     Tests for PredictionResult     #
######################################


def test_prediction_result_from_predictions_returns_prediction_result() -> None:
    result = PredictionResult.from_predictions(
        example_ids=["id1", "id2"],
        predictions=["pred1", "pred2"],
    )
    assert result == PredictionResult(
        records=[
            PredictionRecord(example_id="id1", prediction="pred1"),
            PredictionRecord(example_id="id2", prediction="pred2"),
        ]
    )


def test_prediction_result_from_predictions_returns_prediction_result_with_metadata() -> None:
    result = PredictionResult.from_predictions(
        example_ids=["id1", "id2"], predictions=["pred1", "pred2"], metadata={"tag": "meow"}
    )
    assert result == PredictionResult(
        records=[
            PredictionRecord(example_id="id1", prediction="pred1"),
            PredictionRecord(example_id="id2", prediction="pred2"),
        ],
        metadata={"tag": "meow"},
    )


def test_prediction_result_from_predictions_with_empty_inputs() -> None:
    result = PredictionResult.from_predictions(example_ids=[], predictions=[])
    assert result == PredictionResult(records=[])


def test_prediction_result_from_predictions_with_empty_inputs_with_metadata() -> None:
    result = PredictionResult.from_predictions(
        example_ids=[], predictions=[], metadata={"tag": "meow"}
    )
    assert result == PredictionResult(records=[], metadata={"tag": "meow"})


def test_prediction_result_from_predictions_with_single_pair() -> None:
    result = PredictionResult.from_predictions(
        example_ids=["id1"],
        predictions=["pred1"],
    )
    assert result == PredictionResult(
        records=[PredictionRecord(example_id="id1", prediction="pred1")]
    )


def test_prediction_result_from_predictions_raises_when_lengths_differ() -> None:
    with pytest.raises(ValueError, match="example_ids and predictions must have the same length"):
        PredictionResult.from_predictions(
            example_ids=["id1", "id2"],
            predictions=["pred1"],
        )


def test_prediction_result_from_predictions_raises_when_ids_longer_than_predictions() -> None:
    with pytest.raises(ValueError, match="example_ids and predictions must have the same length"):
        PredictionResult.from_predictions(
            example_ids=["id1", "id2", "id3"],
            predictions=["pred1", "pred2"],
        )


def test_prediction_result_from_predictions_raises_when_predictions_longer_than_ids() -> None:
    with pytest.raises(ValueError, match="example_ids and predictions must have the same length"):
        PredictionResult.from_predictions(
            example_ids=["id1"],
            predictions=["pred1", "pred2"],
        )


def test_prediction_result_from_predictions_with_dict_predictions() -> None:
    predictions = [{"answer": "yes"}, {"answer": "no"}]
    result = PredictionResult.from_predictions(
        example_ids=["id1", "id2"],
        predictions=predictions,
    )
    assert result == PredictionResult(
        records=[
            PredictionRecord(example_id="id1", prediction={"answer": "yes"}),
            PredictionRecord(example_id="id2", prediction={"answer": "no"}),
        ]
    )


def test_prediction_result_from_dict() -> None:
    result = PredictionResult.from_dict({"id1": "pred1", "id2": "pred2"})
    assert result == PredictionResult(
        records=[
            PredictionRecord(example_id="id1", prediction="pred1"),
            PredictionRecord(example_id="id2", prediction="pred2"),
        ]
    )


def test_prediction_result_from_dict_with_dict() -> None:
    result = PredictionResult.from_dict({"id1": {"answer": "yes"}, "id2": {"answer": "no"}})
    assert result == PredictionResult(
        records=[
            PredictionRecord(example_id="id1", prediction={"answer": "yes"}),
            PredictionRecord(example_id="id2", prediction={"answer": "no"}),
        ]
    )


def test_prediction_result_from_dict_with_metadata() -> None:
    result = PredictionResult.from_dict({"id1": "pred1", "id2": "pred2"}, metadata={"tag": "meow"})
    assert result == PredictionResult(
        records=[
            PredictionRecord(example_id="id1", prediction="pred1"),
            PredictionRecord(example_id="id2", prediction="pred2"),
        ],
        metadata={"tag": "meow"},
    )


def test_prediction_result_from_dict_empty() -> None:
    result = PredictionResult.from_dict({})
    assert result == PredictionResult(records=[])


def test_prediction_result_from_dict_empty_with_metadata() -> None:
    result = PredictionResult.from_dict({}, metadata={"tag": "meow"})
    assert result == PredictionResult(records=[], metadata={"tag": "meow"})


def test_prediction_result_to_dict() -> None:
    result = PredictionResult(
        records=[
            PredictionRecord(example_id="id1", prediction="pred1"),
            PredictionRecord(example_id="id2", prediction="pred2"),
        ]
    )
    assert result.to_dict() == {"id1": "pred1", "id2": "pred2"}


def test_prediction_result_to_dict_with_dict() -> None:
    result = PredictionResult(
        records=[
            PredictionRecord(example_id="id1", prediction={"answer": "yes"}),
            PredictionRecord(example_id="id2", prediction={"answer": "no"}),
        ]
    )
    assert result.to_dict() == {"id1": {"answer": "yes"}, "id2": {"answer": "no"}}


def test_prediction_result_to_dict_empty() -> None:
    result = PredictionResult(records=[])
    assert result.to_dict() == {}


def test_prediction_result_metadata_defaults_to_none() -> None:
    result = PredictionResult(records=[])
    assert result.metadata is None


def test_prediction_result_records_field() -> None:
    records = [PredictionRecord(example_id="id1", prediction="pred1")]
    result = PredictionResult(records=records)
    assert result.records == records


def test_prediction_result_from_predictions_metadata_defaults_to_none() -> None:
    result = PredictionResult.from_predictions(example_ids=["id1"], predictions=["pred1"])
    assert result.metadata is None


def test_prediction_result_from_dict_metadata_defaults_to_none() -> None:
    result = PredictionResult.from_dict({"id1": "pred1"})
    assert result.metadata is None


def test_prediction_result_to_dict_preserves_order() -> None:
    result = PredictionResult.from_predictions(
        example_ids=["a", "b", "c"],
        predictions=["x", "y", "z"],
    )
    assert list(result.to_dict().keys()) == ["a", "b", "c"]
    assert list(result.to_dict().values()) == ["x", "y", "z"]


@pytest.mark.parametrize(
    "prediction",
    [
        pytest.param("text", id="str"),
        pytest.param(0, id="zero"),
        pytest.param(None, id="none"),
    ],
)
def test_prediction_result_to_dict_various_prediction_types(prediction: object) -> None:
    result = PredictionResult.from_predictions(
        example_ids=["id1"],
        predictions=[prediction],
    )
    assert result.to_dict() == {"id1": prediction}


def test_prediction_record_repr() -> None:
    record = PredictionRecord(example_id="q1", prediction="4")
    assert repr(record) == "PredictionRecord(example_id='q1', prediction='4', metadata=None)"


def test_prediction_record_repr_with_metadata() -> None:
    record = PredictionRecord(example_id="q1", prediction="4", metadata={"source": "llm"})
    assert repr(record) == (
        "PredictionRecord(example_id='q1', prediction='4', metadata={'source': 'llm'})"
    )


def test_prediction_result_repr_empty() -> None:
    assert repr(PredictionResult(records=[])) == "PredictionResult(records=[], metadata=None)"


def test_prediction_result_repr() -> None:
    result = PredictionResult(records=[PredictionRecord(example_id="id1", prediction="pred1")])
    assert (
        repr(result)
        == "PredictionResult(records=[PredictionRecord(example_id='id1', prediction='pred1', metadata=None)], metadata=None)"
    )


def test_prediction_record_equality() -> None:
    assert PredictionRecord(example_id="q1", prediction="4") == PredictionRecord(
        example_id="q1", prediction="4"
    )


@pytest.mark.parametrize(
    ("rec1", "rec2"),
    [
        pytest.param(
            PredictionRecord(example_id="q1", prediction="4"),
            PredictionRecord(example_id="q2", prediction="4"),
            id="different_example_id",
        ),
        pytest.param(
            PredictionRecord(example_id="q1", prediction="4"),
            PredictionRecord(example_id="q1", prediction="5"),
            id="different_prediction",
        ),
        pytest.param(
            PredictionRecord(example_id="q1", prediction="4", metadata={"a": 1}),
            PredictionRecord(example_id="q1", prediction="4", metadata={"a": 2}),
            id="different_metadata",
        ),
    ],
)
def test_prediction_record_inequality(
    rec1: PredictionRecord, rec2: PredictionRecord
) -> None:
    assert rec1 != rec2
