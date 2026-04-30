r"""Unit tests for PredictionResult.from_predictions."""

import pytest

from argos.meta_agent.benchmark import Benchmark, BenchmarkExample

###############################
#     Tests for Benchmark     #
###############################


def test_benchmark_from_examples() -> None:
    benchmark = Benchmark.from_examples(
        [
            BenchmarkExample(id="id1", input=1, target=1),
            BenchmarkExample(id="id2", input=2, target=1),
            BenchmarkExample(id="id3", input=3, target=0),
            BenchmarkExample(id="id4", input=4, target=1),
            BenchmarkExample(id="id5", input=5, target=0),
        ]
    )
    assert benchmark == Benchmark(
        {
            "id1": BenchmarkExample(id="id1", input=1, target=1),
            "id2": BenchmarkExample(id="id2", input=2, target=1),
            "id3": BenchmarkExample(id="id3", input=3, target=0),
            "id4": BenchmarkExample(id="id4", input=4, target=1),
            "id5": BenchmarkExample(id="id5", input=5, target=0),
        }
    )


def test_benchmark_from_examples_with_metadata() -> None:
    benchmark = Benchmark.from_examples(
        [
            BenchmarkExample(id="id1", input=1, target=1),
            BenchmarkExample(id="id2", input=2, target=1),
            BenchmarkExample(id="id3", input=3, target=0),
            BenchmarkExample(id="id4", input=4, target=1),
            BenchmarkExample(id="id5", input=5, target=0),
        ],
        metadata={"tag": "meow"},
    )
    assert benchmark == Benchmark(
        {
            "id1": BenchmarkExample(id="id1", input=1, target=1),
            "id2": BenchmarkExample(id="id2", input=2, target=1),
            "id3": BenchmarkExample(id="id3", input=3, target=0),
            "id4": BenchmarkExample(id="id4", input=4, target=1),
            "id5": BenchmarkExample(id="id5", input=5, target=0),
        },
        metadata={"tag": "meow"},
    )


def test_benchmark_from_examples_empty() -> None:
    assert Benchmark.from_examples([]) == Benchmark({})


def test_benchmark_from_examples_empty_with_metadata() -> None:
    assert Benchmark.from_examples([], metadata={"tag": "meow"}) == Benchmark(
        {}, metadata={"tag": "meow"}
    )


def test_benchmark_from_examples_with_duplicated_example_ids() -> None:
    with pytest.raises(ValueError, match="Some example IDs are duplicated"):
        Benchmark.from_examples(
            [
                BenchmarkExample(id="id1", input=1, target=1),
                BenchmarkExample(id="id2", input=2, target=1),
                BenchmarkExample(id="id3", input=3, target=0),
                BenchmarkExample(id="id4", input=4, target=1),
                BenchmarkExample(id="id2", input=5, target=0),
            ]
        )
