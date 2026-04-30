r"""Unit tests for BatchPredictor."""

from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableConfig

from argos.meta_agent.agent import BaseAgent
from argos.meta_agent.interface import Benchmark, PredictionRecord, PredictionResult
from argos.meta_agent.predictor import BatchPredictor


@pytest.fixture
def mock_agent() -> BaseAgent:
    return Mock(
        spec=BaseAgent,
        predict=Mock(
            side_effect=[[{"answer": "a1"}, {"answer": "a2"}], [{"answer": "a3"}, {"answer": "a4"}]]
        ),
    )


@pytest.fixture
def mock_benchmark() -> Benchmark:
    return Mock(
        spec=Benchmark,
        examples={
            "id1": {"query": "q1"},
            "id2": {"query": "q2"},
            "id3": {"query": "q3"},
            "id4": {"query": "q4"},
        },
    )


@pytest.fixture
def results() -> PredictionResult:
    return PredictionResult(
        [
            PredictionRecord(example_id="id1", prediction={"answer": "a1"}),
            PredictionRecord(example_id="id2", prediction={"answer": "a2"}),
            PredictionRecord(example_id="id3", prediction={"answer": "a3"}),
            PredictionRecord(example_id="id4", prediction={"answer": "a4"}),
        ]
    )


####################################
#     Tests for BatchPredictor     #
####################################


def test_batch_predictor_default_batch_size() -> None:
    predictor = BatchPredictor()
    assert predictor._batch_size == 1


def test_batch_predictor_custom_batch_size() -> None:
    predictor = BatchPredictor(batch_size=4)
    assert predictor._batch_size == 4


def test_batch_predictor_default_config_uses_batch_size_as_max_concurrency() -> None:
    predictor = BatchPredictor(batch_size=4)
    assert predictor._config["max_concurrency"] == 4


def test_batch_predictor_custom_config_is_stored() -> None:
    config = RunnableConfig(max_concurrency=8)
    predictor = BatchPredictor(batch_size=2, config=config)
    assert predictor._config == config


def test_batch_predictor_none_config_generates_default() -> None:
    assert BatchPredictor()._config is not None


def test_batch_predictor_repr() -> None:
    assert repr(BatchPredictor()).startswith("BatchPredictor(")


def test_batch_predictor_str() -> None:
    assert str(BatchPredictor()).startswith("BatchPredictor(")


def test_batch_predictor_predict_returns_prediction_result_batch_size_1(
    mock_agent: BaseAgent, mock_benchmark: Benchmark, results: PredictionResult
) -> None:
    mock_agent.predict.side_effect = [
        [{"answer": "a1"}],
        [{"answer": "a2"}],
        [{"answer": "a3"}],
        [{"answer": "a4"}],
    ]
    predictor = BatchPredictor(batch_size=1)
    result = predictor.predict(mock_agent, mock_benchmark)
    assert result == results
    assert mock_agent.predict.call_count == 4


def test_batch_predictor_predict_returns_prediction_result_batch_size_2(
    mock_agent: BaseAgent, mock_benchmark: Benchmark, results: PredictionResult
) -> None:
    predictor = BatchPredictor(batch_size=2)
    result = predictor.predict(mock_agent, mock_benchmark)
    assert result == results
    assert mock_agent.predict.call_count == 2


def test_batch_predictor_predict_returns_prediction_result_batch_size_10(
    mock_agent: BaseAgent, mock_benchmark: Benchmark, results: PredictionResult
) -> None:
    mock_agent.predict.side_effect = [
        [{"answer": "a1"}, {"answer": "a2"}, {"answer": "a3"}, {"answer": "a4"}]
    ]
    predictor = BatchPredictor(batch_size=10)
    result = predictor.predict(mock_agent, mock_benchmark)
    assert result == results
    assert mock_agent.predict.call_count == 1


def test_batch_predictor_predict_passes_config_to_agent(
    mock_agent: BaseAgent, mock_benchmark: Benchmark
) -> None:
    config = RunnableConfig(max_concurrency=4)
    predictor = BatchPredictor(batch_size=2, config=config)
    predictor.predict(mock_agent, mock_benchmark)
    for call in mock_agent.predict.call_args_list:
        assert call.kwargs["config"] is config


def test_batch_predictor_predict_with_empty_benchmark(mock_agent: BaseAgent) -> None:
    benchmark = Benchmark(examples={})
    predictor = BatchPredictor(batch_size=2)
    result = predictor.predict(mock_agent, benchmark)
    assert result == PredictionResult([])
    mock_agent.predict.assert_not_called()
