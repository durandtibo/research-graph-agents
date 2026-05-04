r"""Unit tests for BatchPredictor."""

from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableConfig

from argos.meta_agent.agents import Agent, BaseAgent
from argos.meta_agent.benchmark import Benchmark, BenchmarkExample
from argos.meta_agent.prediction import (
    PredictionRecord,
    PredictionResult,
)
from argos.meta_agent.predictors import BasePredictor, BatchPredictor
from tests.unit.helpers.runnable import DoubleRunnable


@pytest.fixture
def benchmark() -> Benchmark:
    return Benchmark(
        {
            "id1": BenchmarkExample(id="id1", input=1, target=None),
            "id2": BenchmarkExample(id="id2", input=2, target=None),
            "id3": BenchmarkExample(id="id3", input=3, target=None),
            "id4": BenchmarkExample(id="id4", input=4, target=None),
            "id5": BenchmarkExample(id="id5", input=5, target=None),
        }
    )


@pytest.fixture
def mock_agent() -> BaseAgent:
    return Mock(
        spec=BaseAgent,
        predict=Mock(
            side_effect=[[{"answer": "a1"}, {"answer": "a2"}], [{"answer": "a3"}, {"answer": "a4"}]]
        ),
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
    assert repr(BatchPredictor()) == (
        "BatchPredictor(\n  (batch_size): 1\n  (config): {'max_concurrency': 1}\n)"
    )


def test_batch_predictor_repr_custom_batch_size() -> None:
    assert repr(BatchPredictor(batch_size=4)) == (
        "BatchPredictor(\n  (batch_size): 4\n  (config): {'max_concurrency': 4}\n)"
    )


def test_batch_predictor_str() -> None:
    assert str(BatchPredictor()) == (
        "BatchPredictor(\n  (batch_size): 1\n  (config): {'max_concurrency': 1}\n)"
    )


def test_batch_predictor_str_custom_batch_size() -> None:
    assert str(BatchPredictor(batch_size=4)) == (
        "BatchPredictor(\n  (batch_size): 4\n  (config): {'max_concurrency': 4}\n)"
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4, 10])
def test_batch_predictor_predict_batch_size(batch_size: int, benchmark: Benchmark) -> None:
    predictor = BatchPredictor(batch_size=batch_size)
    result = predictor.predict(agent=Agent(DoubleRunnable()), benchmark=benchmark)
    assert result == PredictionResult(
        [
            PredictionRecord(example_id="id1", prediction=2),
            PredictionRecord(example_id="id2", prediction=4),
            PredictionRecord(example_id="id3", prediction=6),
            PredictionRecord(example_id="id4", prediction=8),
            PredictionRecord(example_id="id5", prediction=10),
        ]
    )


def test_batch_predictor_predict_passes_config_to_agent(benchmark: Benchmark) -> None:
    agent = Mock(spec=BaseAgent, predict=Mock(side_effect=[[2, 4], [6, 8], [10]]))
    config = RunnableConfig(max_concurrency=4)
    predictor = BatchPredictor(batch_size=2, config=config)
    predictor.predict(agent, benchmark)
    for call in agent.predict.call_args_list:
        assert call.kwargs["config"] is config


def test_batch_predictor_predict_with_empty_benchmark() -> None:
    predictor = BatchPredictor(batch_size=2)
    result = predictor.predict(agent=Agent(DoubleRunnable()), benchmark=Benchmark(examples={}))
    assert result == PredictionResult([])


def test_batch_predictor_is_instance_of_base_predictor() -> None:
    assert isinstance(BatchPredictor(), BasePredictor)


def test_batch_predictor_predict_single_example() -> None:
    agent = Agent(DoubleRunnable())
    benchmark = Benchmark({"id1": BenchmarkExample(id="id1", input=7, target=None)})
    result = BatchPredictor(batch_size=4).predict(agent=agent, benchmark=benchmark)
    assert result == PredictionResult([PredictionRecord(example_id="id1", prediction=14)])


def test_batch_predictor_predict_with_batch_size_larger_than_examples(
    benchmark: Benchmark,
) -> None:
    predictor = BatchPredictor(batch_size=100)
    result = predictor.predict(agent=Agent(DoubleRunnable()), benchmark=benchmark)
    assert result == PredictionResult(
        [
            PredictionRecord(example_id="id1", prediction=2),
            PredictionRecord(example_id="id2", prediction=4),
            PredictionRecord(example_id="id3", prediction=6),
            PredictionRecord(example_id="id4", prediction=8),
            PredictionRecord(example_id="id5", prediction=10),
        ]
    )
