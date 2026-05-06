from __future__ import annotations

import polars as pl
import pytest

from argos.meta_agent.benchmark import Benchmark, BenchmarkExample
from argos.meta_agent.evaluators import BaseEvaluator, NoOpEvaluator
from argos.meta_agent.prediction import PredictionRecord, PredictionResult


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["id1", "id2", "id3", "id4", "id5"],
            "input": ["input1", "input2", "input3", "input4", "input5"],
            "target": ["target1", "target2", "target3", "target4", "target5"],
            "metadata": [
                {"tag": "tag1"},
                {"tag": "tag2"},
                {"tag": "tag3"},
                {"tag": "tag4"},
                {"tag": "tag5"},
            ],
        },
        schema=pl.Schema(
            {
                "id": pl.String,
                "input": pl.String,
                "target": pl.String,
                "metadata": pl.Struct({"tag": pl.String}),
            },
        ),
    )


###################################
#     Tests for NoOpEvaluator     #
###################################


def test_noop_evaluator_repr() -> None:
    assert repr(NoOpEvaluator()) == "NoOpEvaluator()"


def test_noop_evaluator_str() -> None:
    assert str(NoOpEvaluator()) == "NoOpEvaluator()"


def test_noop_evaluator_evaluate(dataframe: pl.DataFrame) -> None:
    assert NoOpEvaluator().evaluate(dataframe).equal(Result({}))


def test_noop_evaluator_evaluate_empty() -> None:
    assert NoOpEvaluator().evaluate(PredictionResult([]), Benchmark({})) == {}


def test_noop_evaluator_returns_empty_dict_regardless_of_predictions(
    benchmark: Benchmark,
) -> None:
    predictions = PredictionResult([PredictionRecord(example_id="id1", prediction="unexpected")])
    assert NoOpEvaluator().evaluate(predictions=predictions, benchmark=benchmark) == {}


def test_noop_evaluator_is_instance_of_base_evaluator() -> None:
    assert isinstance(NoOpEvaluator(), BaseEvaluator)
