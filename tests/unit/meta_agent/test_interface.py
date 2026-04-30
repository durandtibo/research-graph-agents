r"""Unit tests for PredictionResult.from_predictions."""

import pytest

from argos.meta_agent.interface import PredictionRecord, PredictionResult

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


def test_prediction_result_from_predictions_with_empty_inputs() -> None:
    result = PredictionResult.from_predictions(example_ids=[], predictions=[])
    assert result == PredictionResult(records=[])


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


def test_prediction_result_from_dict_empty() -> None:
    result = PredictionResult.from_dict({})
    assert result == PredictionResult(records=[])


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
