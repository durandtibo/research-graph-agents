from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import polars as pl
import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from polars.testing import assert_frame_equal

from argos.metrics import BinaryClassificationResults
from argos.nodes.haiku_judge import HaikuJudgeResult
from argos.tasks.autoprompt.haiku_judge import (
    ExperimentConfig,
    create_graph,
    evaluate_metrics,
    prepare_dataset,
    prepare_results,
    run_experiment,
    run_inference,
    run_inference_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

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
def mock_graph(mock_outputs: list[dict[str, Any]]) -> CompiledStateGraph:
    graph = Mock(spec=CompiledStateGraph)
    graph.batch.side_effect = [mock_outputs]
    return graph


@pytest.fixture
def mock_outputs() -> list[dict[str, Any]]:
    return [
        {
            "topic": "rain",
            "haiku": (
                "Gray sky descends slow,\n"
                "Cool drops kiss the thirsty ground,\n"
                "Silence finds the leaf."
            ),
            "evaluation": HaikuJudgeResult(
                structure_passed=True, topic_passed=True, score=10, reasoning="reason1", passed=True
            ),
        },
        {
            "topic": "cat",
            "haiku": (
                "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
            ),
            "evaluation": HaikuJudgeResult(
                structure_passed=True, topic_passed=True, score=9, reasoning="reason2", passed=True
            ),
        },
        {
            "topic": "mountain",
            "haiku": (
                "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
            ),
            "evaluation": HaikuJudgeResult(
                structure_passed=True, topic_passed=True, score=8, reasoning="reason3", passed=True
            ),
        },
    ]


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
                "score": 10,
                "passed": True,
                "target": True,
                "structure_passed": True,
                "structure_target": True,
                "topic_passed": True,
                "topic_target": True,
                "reasoning": "reason1",
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                "score": 9,
                "passed": True,
                "target": True,
                "structure_passed": True,
                "structure_target": True,
                "topic_passed": True,
                "topic_target": True,
                "reasoning": "reason2",
            },
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "score": 8,
                "passed": True,
                "target": True,
                "structure_passed": True,
                "structure_target": True,
                "topic_passed": True,
                "topic_target": True,
                "reasoning": "reason3",
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
        assert "judge" in graph.nodes


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


#####################################
#     Tests for prepare_results     #
#####################################


def test_prepare_results_returns_dataframe(
    mock_dataset: pl.DataFrame, mock_outputs: list, mock_results: pl.DataFrame
) -> None:
    assert_frame_equal(prepare_results(dataset=mock_dataset, outputs=mock_outputs), mock_results)


####################################
#     Tests for run_experiment     #
####################################


def test_run_experiment_results_file_does_not_exist(
    tmp_path: Path, mock_results: pl.DataFrame
) -> None:
    path_results = tmp_path.joinpath("data").joinpath("results.parquet")

    def fake_run_inference(
        model: str,  # noqa: ARG001
        system_prompt: str,  # noqa: ARG001
        path_results: Path,
        batch_size: int = 20,  # noqa: ARG001
    ) -> None:
        path_results.parent.mkdir(parents=True, exist_ok=True)
        mock_results.write_parquet(path_results)

    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="You are a haiku judge",
        path_experiment=path_results.parent,
    )
    with patch(f"{MODULE}.run_inference", side_effect=fake_run_inference) as run_inference_mock:
        metrics = run_experiment(config)
    run_inference_mock.assert_called_once_with(
        model="gpt-4o", system_prompt="You are a haiku judge", path_results=path_results
    )
    assert metrics == {
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


def test_run_experiment_results_file_exists(tmp_path: Path, mock_results: pl.DataFrame) -> None:
    path_results = tmp_path.joinpath("data").joinpath("results.parquet")
    path_results.parent.mkdir(parents=True, exist_ok=True)
    mock_results.write_parquet(path_results)

    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="You are a haiku judge",
        path_experiment=path_results.parent,
    )
    with patch(f"{MODULE}.run_inference") as run_inference_mock:
        metrics = run_experiment(config)
    run_inference_mock.assert_not_called()
    assert metrics == {
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


###################################
#     Tests for run_inference     #
###################################


def test_run_inference(
    tmp_path: Path,
    mock_dataset: pl.DataFrame,
    mock_graph: CompiledStateGraph,
    mock_results: pl.DataFrame,
) -> None:
    path_results = tmp_path.joinpath("data").joinpath("results.parquet")
    with (
        patch(f"{MODULE}.create_graph", return_value=mock_graph) as create_graph_mock,
        patch(
            f"{MODULE}.generate_haiku_dataset", return_value=mock_dataset
        ) as generate_haiku_dataset_mock,
        patch(
            f"{MODULE}.run_inference_pipeline", return_value=mock_results
        ) as run_inference_pipeline_mock,
    ):
        result = run_inference(
            model="gpt-4o", system_prompt="You are a haiku judge", path_results=path_results
        )
        assert_frame_equal(result, mock_results)
        create_graph_mock.assert_called_once_with(
            model="gpt-4o", system_prompt="You are a haiku judge"
        )
        generate_haiku_dataset_mock.assert_called_once_with()
        run_inference_pipeline_mock.assert_called_once_with(
            dataset=mock_dataset, graph=mock_graph, batch_size=20
        )

    assert path_results.is_file()
    assert_frame_equal(pl.read_parquet(path_results), mock_results)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_run_inference_batch_size(
    tmp_path: Path,
    mock_dataset: pl.DataFrame,
    mock_graph: CompiledStateGraph,
    mock_results: pl.DataFrame,
    batch_size: int,
) -> None:
    path_results = tmp_path.joinpath("data").joinpath("results.parquet")
    with (
        patch(f"{MODULE}.create_graph", return_value=mock_graph) as create_graph_mock,
        patch(
            f"{MODULE}.generate_haiku_dataset", return_value=mock_dataset
        ) as generate_haiku_dataset_mock,
        patch(
            f"{MODULE}.run_inference_pipeline", return_value=mock_results
        ) as run_inference_pipeline_mock,
    ):
        result = run_inference(
            model="gpt-4o",
            system_prompt="You are a haiku judge",
            path_results=path_results,
            batch_size=batch_size,
        )
        assert_frame_equal(result, mock_results)
        assert path_results.is_file()
        create_graph_mock.assert_called_once_with(
            model="gpt-4o", system_prompt="You are a haiku judge"
        )
        generate_haiku_dataset_mock.assert_called_once_with()
        run_inference_pipeline_mock.assert_called_once_with(
            dataset=mock_dataset, graph=mock_graph, batch_size=batch_size
        )


############################################
#     Tests for run_inference_pipeline     #
############################################


def test_run_inference_pipeline(
    mock_dataset: pl.DataFrame, mock_graph: CompiledStateGraph, mock_results: pl.DataFrame
) -> None:
    result = run_inference_pipeline(dataset=mock_dataset, graph=mock_graph)
    assert_frame_equal(result, mock_results)


def test_run_inference_pipeline_batch_size_1(
    mock_dataset: pl.DataFrame, mock_results: pl.DataFrame
) -> None:
    mock_graph = Mock(spec=CompiledStateGraph)
    mock_graph.batch.side_effect = [
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=10,
                    reasoning="reason1",
                    passed=True,
                ),
            }
        ],
        [
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=9,
                    reasoning="reason2",
                    passed=True,
                ),
            },
        ],
        [
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=8,
                    reasoning="reason3",
                    passed=True,
                ),
            },
        ],
    ]
    result = run_inference_pipeline(dataset=mock_dataset, graph=mock_graph, batch_size=1)
    assert_frame_equal(result, mock_results)


def test_run_inference_pipeline_batch_size_2(
    mock_dataset: pl.DataFrame, mock_results: pl.DataFrame
) -> None:
    mock_graph = Mock(spec=CompiledStateGraph)
    mock_graph.batch.side_effect = [
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=10,
                    reasoning="reason1",
                    passed=True,
                ),
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=9,
                    reasoning="reason2",
                    passed=True,
                ),
            },
        ],
        [
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "evaluation": HaikuJudgeResult(
                    structure_passed=True,
                    topic_passed=True,
                    score=8,
                    reasoning="reason3",
                    passed=True,
                ),
            },
        ],
    ]
    result = run_inference_pipeline(dataset=mock_dataset, graph=mock_graph, batch_size=2)
    assert_frame_equal(result, mock_results)
