r"""Unit tests for BatchPredictor."""

from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableConfig

from argos.meta_agent.agents import Agent, BaseAgent
from argos.meta_agent.batches import Batch
from argos.meta_agent.entities import LabeledExample, Prediction
from argos.meta_agent.predictors import BasePredictor, BatchPredictor
from tests.unit.helpers.runnable import DoubleRunnable


@pytest.fixture
def dataset() -> Batch:
    return Batch(
        {
            "id1": LabeledExample(id="id1", input=1, target=None),
            "id2": LabeledExample(id="id2", input=2, target=None),
            "id3": LabeledExample(id="id3", input=3, target=None),
            "id4": LabeledExample(id="id4", input=4, target=None),
            "id5": LabeledExample(id="id5", input=5, target=None),
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


def test_batch_predictor_repr() -> None:
    assert repr(BatchPredictor()) == (
        "BatchPredictor(\n  (batch_size): 1\n  (config): {'max_concurrency': 1}\n)"
    )


def test_batch_predictor_repr_custom_batch_size() -> None:
    assert repr(BatchPredictor(batch_size=4)) == (
        "BatchPredictor(\n  (batch_size): 4\n  (config): {'max_concurrency': 4}\n)"
    )


def test_batch_predictor_custom_config_repr() -> None:
    config = RunnableConfig(max_concurrency=8)
    predictor = BatchPredictor(batch_size=2, config=config)
    assert repr(predictor) == (
        "BatchPredictor(\n  (batch_size): 2\n  (config): {'max_concurrency': 8}\n)"
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
def test_batch_predictor_predict_batch_size(batch_size: int, dataset: Batch) -> None:
    predictor = BatchPredictor(batch_size=batch_size)
    result = predictor.predict(agent=Agent(DoubleRunnable()), dataset=dataset)
    assert result == Batch(
        {
            "id1": Prediction(id="id1", prediction=2),
            "id2": Prediction(id="id2", prediction=4),
            "id3": Prediction(id="id3", prediction=6),
            "id4": Prediction(id="id4", prediction=8),
            "id5": Prediction(id="id5", prediction=10),
        }
    )


def test_batch_predictor_predict_passes_config_to_agent(dataset: Batch) -> None:
    agent = Mock(spec=BaseAgent, predict=Mock(side_effect=[[2, 4], [6, 8], [10]]))
    config = RunnableConfig(max_concurrency=4)
    predictor = BatchPredictor(batch_size=2, config=config)
    predictor.predict(agent, dataset)
    for call in agent.predict.call_args_list:
        assert call.kwargs["config"] is config


def test_batch_predictor_predict_with_empty_dataset() -> None:
    predictor = BatchPredictor(batch_size=2)
    result = predictor.predict(agent=Agent(DoubleRunnable()), dataset=Batch({}))
    assert result == Batch({})


def test_batch_predictor_is_instance_of_base_predictor() -> None:
    assert isinstance(BatchPredictor(), BasePredictor)


def test_batch_predictor_predict_single_example() -> None:
    agent = Agent(DoubleRunnable())
    dataset = Batch({"id1": LabeledExample(id="id1", input=7, target=None)})
    result = BatchPredictor(batch_size=4).predict(agent=agent, dataset=dataset)
    assert result == Batch({"id1": Prediction(id="id1", prediction=14)})


def test_batch_predictor_predict_with_batch_size_larger_than_examples(
    dataset: Batch,
) -> None:
    predictor = BatchPredictor(batch_size=100)
    result = predictor.predict(agent=Agent(DoubleRunnable()), dataset=dataset)
    assert result == Batch(
        {
            "id1": Prediction(id="id1", prediction=2),
            "id2": Prediction(id="id2", prediction=4),
            "id3": Prediction(id="id3", prediction=6),
            "id4": Prediction(id="id4", prediction=8),
            "id5": Prediction(id="id5", prediction=10),
        }
    )
