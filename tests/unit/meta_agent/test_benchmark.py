r"""Unit tests for Benchmark and BenchmarkExample."""

import pytest

from argos.meta_agent.benchmark import Benchmark, BenchmarkExample

#####################################
#     Tests for BenchmarkExample     #
#####################################


def test_benchmark_example_id() -> None:
    assert BenchmarkExample(id="q1", input="What is 2+2?", target="4").id == "q1"


def test_benchmark_example_input() -> None:
    assert BenchmarkExample(id="q1", input="What is 2+2?", target="4").input == "What is 2+2?"


def test_benchmark_example_target() -> None:
    assert BenchmarkExample(id="q1", input="What is 2+2?", target="4").target == "4"


def test_benchmark_example_metadata_defaults_to_none() -> None:
    assert BenchmarkExample(id="q1", input="What is 2+2?", target="4").metadata is None


def test_benchmark_example_metadata_custom() -> None:
    example = BenchmarkExample(id="q1", input="What is 2+2?", target="4", metadata={"src": "math"})
    assert example.metadata == {"src": "math"}


@pytest.mark.parametrize(
    ("inp", "target"),
    [
        pytest.param(1, 1, id="int"),
        pytest.param(1.5, 2.5, id="float"),
        pytest.param({"key": "val"}, {"out": "val"}, id="dict"),
    ],
)
def test_benchmark_example_supports_non_string_types(inp: object, target: object) -> None:
    example = BenchmarkExample(id="q1", input=inp, target=target)
    assert example.input == inp
    assert example.target == target


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
