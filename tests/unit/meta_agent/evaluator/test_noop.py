from __future__ import annotations

import pytest

from argos.meta_agent.benchmark import Benchmark, BenchmarkExample
from argos.meta_agent.evaluator import NoOpEvaluator
from argos.meta_agent.prediction import PredictionRecord, PredictionResult


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
def predictions() -> PredictionResult:
    return PredictionResult(
        [
            PredictionRecord(example_id="id1", prediction=2),
            PredictionRecord(example_id="id2", prediction=4),
            PredictionRecord(example_id="id3", prediction=6),
            PredictionRecord(example_id="id4", prediction=8),
            PredictionRecord(example_id="id5", prediction=10),
        ]
    )


###################################
#     Tests for NoOpEvaluator     #
###################################


def test_noop_evaluator_repr() -> None:
    assert repr(NoOpEvaluator()) == "NoOpEvaluator()"


def test_noop_evaluator_str() -> None:
    assert str(NoOpEvaluator()) == "NoOpEvaluator()"


def test_noop_evaluator_evaluate(benchmark: Benchmark, predictions: PredictionResult) -> None:
    assert NoOpEvaluator().evaluate(predictions=predictions, benchmark=benchmark) == {}


def test_noop_evaluator_evaluate_empty() -> None:
    assert NoOpEvaluator().evaluate(PredictionResult([]), Benchmark({})) == {}
