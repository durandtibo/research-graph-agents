from __future__ import annotations

from unittest.mock import Mock, patch

import polars as pl
import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from polars.testing import assert_frame_equal

from argos.metrics import BinaryClassificationResults
from argos.tasks.autoprompt.haiku_judge import (
    create_graph,
    evaluate_metrics,
    prepare_dataset,
)

MODULE = "argos.tasks.autoprompt.haiku_judge"


@pytest.fixture
def mock_dataset() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
        ]
    )


@pytest.fixture
def mock_llm() -> BaseChatModel:
    llm = Mock(spec=BaseChatModel)
    llm.model = "gpt-4o"
    llm.temperature = 0
    return llm


@pytest.fixture
def mock_results() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
                "passed": True,
                "structure_passed": True,
                "topic_passed": True,
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
                "passed": True,
                "structure_passed": True,
                "topic_passed": True,
            },
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
                "passed": True,
                "structure_passed": True,
                "topic_passed": True,
            },
        ]
    )


##################################
#     Tests for create_graph     #
##################################


def test_create_graph_returns_compiled_state_graph(mock_llm: BaseChatModel) -> None:
    """create_graph should return a CompiledStateGraph instance."""
    with patch(f"{MODULE}.init_chat_model", return_value=mock_llm):
        graph = create_graph("gpt-4o", "You are a haiku judge.")
        assert isinstance(graph, CompiledStateGraph)


def test_create_graph_init_chat_model_called_with_correct_model(mock_llm: BaseChatModel) -> None:
    """init_chat_model must be called with the model name passed to
    create_graph."""
    with patch(f"{MODULE}.init_chat_model", return_value=mock_llm) as mock_init:
        create_graph("gpt-4o", "You are a haiku judge.")
        mock_init.assert_called_once_with(model="gpt-4o", temperature=0, max_retries=9999)


def test_create_graph_make_haiku_judge_node_receives_llm_and_system_prompt(
    mock_llm: BaseChatModel,
) -> None:
    """make_haiku_judge_node must be called with the LLM and the judge
    system prompt."""
    with (
        patch(f"{MODULE}.init_chat_model", return_value=mock_llm),
        patch(f"{MODULE}.make_haiku_judge_node") as mock_node_factory,
    ):
        prompt = "You are a strict haiku judge."
        create_graph("gpt-4o", prompt)
        mock_node_factory.assert_called_once_with(mock_llm, system_prompt=prompt)


def test_create_graph_multiple_calls_return_distinct_graphs(mock_llm: BaseChatModel) -> None:
    """Each invocation must return a new, independent
    CompiledStateGraph."""
    with patch(f"{MODULE}.init_chat_model", return_value=mock_llm):
        graph_a = create_graph("gpt-4o", "prompt A")
        graph_b = create_graph("gpt-4o", "prompt B")
        assert graph_a is not graph_b


######################################
#     Tests for evaluate_metrics     #
######################################


def test_evaluate_metrics(mock_results: pl.DataFrame) -> None:
    assert evaluate_metrics(mock_results) == {
        "overall": BinaryClassificationResults(
            n_samples=3,
            accuracy=1.0,
            tp=3,
            tn=0,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            specificity=0.0,
        ),
        "structure": BinaryClassificationResults(
            n_samples=3,
            accuracy=1.0,
            tp=3,
            tn=0,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            specificity=0.0,
        ),
        "topic": BinaryClassificationResults(
            n_samples=3,
            accuracy=1.0,
            tp=3,
            tn=0,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            specificity=0.0,
        ),
    }


def test_evaluate_metrics_mixed_results() -> None:
    assert evaluate_metrics(
        pl.from_dicts(
            [
                {
                    "structure_target": True,
                    "topic_target": True,
                    "target": True,
                    "passed": False,
                    "structure_passed": True,
                    "topic_passed": True,
                },
                {
                    "structure_target": True,
                    "topic_target": True,
                    "target": True,
                    "passed": False,
                    "structure_passed": False,
                    "topic_passed": True,
                },
                {
                    "structure_target": True,
                    "topic_target": True,
                    "target": True,
                    "passed": False,
                    "structure_passed": False,
                    "topic_passed": False,
                },
                {
                    "structure_target": True,
                    "topic_target": True,
                    "target": True,
                    "passed": True,
                    "structure_passed": True,
                    "topic_passed": True,
                },
            ]
        )
    ) == {
        "overall": BinaryClassificationResults(
            n_samples=4,
            accuracy=0.25,
            tp=1,
            tn=0,
            fp=0,
            fn=3,
            precision=1.0,
            recall=0.25,
            f1_score=0.4,
            specificity=0.0,
        ),
        "structure": BinaryClassificationResults(
            n_samples=4,
            accuracy=0.5,
            tp=2,
            tn=0,
            fp=0,
            fn=2,
            precision=1.0,
            recall=0.5,
            f1_score=pytest.approx(2.0 / 3.0, abs=1e-6),
            specificity=0.0,
        ),
        "topic": BinaryClassificationResults(
            n_samples=4,
            accuracy=0.75,
            tp=3,
            tn=0,
            fp=0,
            fn=1,
            precision=1.0,
            recall=0.75,
            f1_score=pytest.approx(1.5 / 1.75, abs=1e-6),
            specificity=0.0,
        ),
    }


#####################################
#     Tests for prepare_dataset     #
#####################################


def test_prepare_dataset_returns_dataframe(mock_dataset: pl.DataFrame) -> None:
    with patch(f"{MODULE}.generate_haiku_dataset", return_value=mock_dataset):
        assert_frame_equal(prepare_dataset(), mock_dataset)
