r"""Unit tests for PredictionResult.from_predictions."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from argos.meta_agent.datasets import Dataset
from argos.meta_agent.examples import BaseExample, Example


@pytest.fixture
def dataset() -> Dataset:
    return Dataset(
        {
            "id1": Example(id="id1", input="input1", target="target1", metadata={"tag": "tag1"}),
            "id2": Example(id="id2", input="input2", target="target2", metadata={"tag": "tag2"}),
            "id3": Example(id="id3", input="input3", target="target3", metadata={"tag": "tag3"}),
            "id4": Example(id="id4", input="input4", target="target4", metadata={"tag": "tag4"}),
            "id5": Example(id="id5", input="input5", target="target5", metadata={"tag": "tag5"}),
        }
    )


@pytest.fixture
def dataset_with_metadata(dataset: Dataset) -> Dataset:
    dataset.metadata = {"tag": "meow"}
    return dataset


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


@pytest.fixture
def examples() -> list[BaseExample]:
    return [
        Example(id="id1", input="input1", target="target1", metadata={"tag": "tag1"}),
        Example(id="id2", input="input2", target="target2", metadata={"tag": "tag2"}),
        Example(id="id3", input="input3", target="target3", metadata={"tag": "tag3"}),
        Example(id="id4", input="input4", target="target4", metadata={"tag": "tag4"}),
        Example(id="id5", input="input5", target="target5", metadata={"tag": "tag5"}),
    ]


#############################
#     Tests for Dataset     #
#############################


def test_dataset_examples() -> None:
    assert Dataset(
        examples={"q1": Example(id="q1", input="What is 2+2?", target="4")}
    ).examples == {"q1": Example(id="q1", input="What is 2+2?", target="4")}


def test_dataset_metadata_() -> None:
    assert Dataset(
        examples={"q1": Example(id="q1", input="What is 2+2?", target="4")},
        metadata={"source": "math"},
    ).metadata == {"source": "math"}


def test_dataset_metadata_default_none() -> None:
    assert (
        Dataset(examples={"q1": Example(id="q1", input="What is 2+2?", target="4")}).metadata
        is None
    )


def test_dataset_equal_true() -> None:
    dataset = Dataset(examples={"q1": Example(id="q1", input="What is 2+2?", target="4")})
    assert dataset.equal(
        Dataset(examples={"q1": Example(id="q1", input="What is 2+2?", target="4")})
    )


def test_dataset_equal_true_empty() -> None:
    assert Dataset(examples={}).equal(Dataset(examples={}))


def test_dataset_equal_true_with_metadata() -> None:
    dataset = Dataset(
        examples={"q1": Example(id="q1", input="What is 2+2?", target="4")},
        metadata={"source": "math"},
    )
    assert dataset.equal(
        Dataset(
            examples={"q1": Example(id="q1", input="What is 2+2?", target="4")},
            metadata={"source": "math"},
        )
    )


def test_dataset_equal_false_different_examples() -> None:
    assert not Dataset(examples={"q1": Example(id="q1", input="What is 2+2?", target="4")}).equal(
        Dataset(examples={"q1": Example(id="q1", input="What is 2+2?", target="5")})
    )


def test_dataset_equal_false_different_example_ids() -> None:
    assert not Dataset(examples={"q1": Example(id="q1", input="What is 2+2?", target="4")}).equal(
        Dataset(examples={"q2": Example(id="q2", input="What is 2+2?", target="4")})
    )


def test_dataset_equal_false_different_number_of_examples() -> None:
    assert not Dataset(examples={"q1": Example(id="q1", input="What is 2+2?", target="4")}).equal(
        Dataset(
            examples={
                "q1": Example(id="q1", input="What is 2+2?", target="4"),
                "q2": Example(id="q2", input="What is 4+2?", target="6"),
            }
        )
    )


def test_dataset_equal_false_different_metadata() -> None:
    examples = {"q1": Example(id="q1", input="What is 2+2?", target="4")}
    assert not Dataset(examples=examples, metadata={"source": "math"}).equal(
        Dataset(examples=examples, metadata={"source": "science"})
    )


def test_dataset_equal_false_metadata_vs_none() -> None:
    examples = {"q1": Example(id="q1", input="What is 2+2?", target="4")}
    assert not Dataset(examples=examples, metadata={"source": "math"}).equal(
        Dataset(examples=examples, metadata=None)
    )


def test_dataset_equal_false_different_type() -> None:
    assert not Dataset(examples={}).equal({})


def test_dataset_equal_nan_false_by_default() -> None:
    assert not Dataset(examples={"q1": Example(id="q1", input=float("nan"), target="4")}).equal(
        Dataset(examples={"q1": Example(id="q1", input=float("nan"), target="4")})
    )


def test_dataset_equal_nan_true() -> None:
    assert Dataset(examples={"q1": Example(id="q1", input=float("nan"), target="4")}).equal(
        Dataset(examples={"q1": Example(id="q1", input=float("nan"), target="4")}),
        equal_nan=True,
    )


def test_dataset_equal_nan_false_by_default_in_metadata() -> None:
    examples = {"q1": Example(id="q1", input="What is 2+2?", target="4")}
    assert not Dataset(examples=examples, metadata={"score": float("nan")}).equal(
        Dataset(examples=examples, metadata={"score": float("nan")})
    )


def test_dataset_equal_nan_true_in_metadata() -> None:
    examples = {"q1": Example(id="q1", input="What is 2+2?", target="4")}
    assert Dataset(examples=examples, metadata={"score": float("nan")}).equal(
        Dataset(examples=examples, metadata={"score": float("nan")}),
        equal_nan=True,
    )


def test_dataset_to_dataframe(dataset: Dataset, dataframe: pl.DataFrame) -> None:
    assert_frame_equal(dataset.to_dataframe(), dataframe)


def test_dataset_to_dataframe_empty() -> None:
    assert_frame_equal(
        Dataset({}).to_dataframe(),
        pl.DataFrame({}),
    )


def test_dataset_from_dataframe(dataset: Dataset, dataframe: pl.DataFrame) -> None:
    assert dataset == Dataset.from_dataframe(dataframe)


def test_dataset_from_dataframe_with_metadata(dataframe: pl.DataFrame) -> None:
    dataset = Dataset.from_dataframe(dataframe, metadata={"tag": "meow"})
    assert dataset.metadata == {"tag": "meow"}


def test_dataset_from_dataframe_with_custom_example_type(dataframe: pl.DataFrame) -> None:
    class CustomExample(Example): ...

    dataset = Dataset.from_dataframe(dataframe, example_type=CustomExample)
    assert all(type(ex) is CustomExample for ex in dataset.examples.values())


def test_dataset_from_dataframe_empty() -> None:
    assert Dataset.from_dataframe(pl.DataFrame({})) == Dataset({})


def test_dataset_from_examples(dataset: Dataset, examples: list[BaseExample]) -> None:
    assert dataset == Dataset.from_examples(examples)


def test_dataset_from_examples_with_metadata(
    dataset_with_metadata: Dataset, examples: list[BaseExample]
) -> None:
    assert dataset_with_metadata == Dataset.from_examples(
        examples,
        metadata={"tag": "meow"},
    )


def test_dataset_from_examples_empty() -> None:
    assert Dataset.from_examples([]) == Dataset({})


def test_dataset_from_examples_empty_with_metadata() -> None:
    assert Dataset.from_examples([], metadata={"tag": "meow"}) == Dataset(
        {}, metadata={"tag": "meow"}
    )


def test_dataset_from_examples_with_duplicated_example_ids() -> None:
    with pytest.raises(ValueError, match="Some example IDs are duplicated"):
        Dataset.from_examples(
            [
                Example(id="id1", input="input1", target="target1", metadata={"tag": "tag1"}),
                Example(id="id2", input="input2", target="target2", metadata={"tag": "tag2"}),
                Example(id="id2", input="input3", target="target3", metadata={"tag": "tag3"}),
            ]
        )


def test_dataset_metadata_defaults_to_none() -> None:
    dataset = Dataset(examples={})
    assert dataset.metadata is None


def test_dataset_from_examples_metadata_defaults_to_none() -> None:
    dataset = Dataset.from_examples([Example(id="id1", input="input1", target="target1")])
    assert dataset.metadata is None


def test_dataset_examples_are_indexed_by_id() -> None:
    ex = Example(id="id1", input="input1", target="target1")
    dataset = Dataset.from_examples([ex])
    assert dataset.examples["id1"] == ex


def test_dataset_repr_empty() -> None:
    assert repr(Dataset(examples={})) == "Dataset(examples={}, metadata=None)"


def test_dataset_repr() -> None:
    ex = Example(id="q1", input="What is 2+2?", target="4")
    assert repr(Dataset(examples={"q1": ex})) == (
        "Dataset(examples={'q1': Example(id='q1', input='What is 2+2?', target='4', metadata=None)}, metadata=None)"
    )


def test_dataset_equality() -> None:
    assert Dataset(examples={"q1": Example(id="q1", input="What is 2+2?", target="4")}) == Dataset(
        examples={"q1": Example(id="q1", input="What is 2+2?", target="4")}
    )


def test_dataset_inequality_different_examples() -> None:
    assert Dataset(examples={"q1": Example(id="q1", input="What is 2+2?", target="4")}) != Dataset(
        examples={"q1": Example(id="q1", input="What is 2+2?", target="5")}
    )
