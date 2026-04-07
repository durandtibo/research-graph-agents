r"""Contain code to run the autoprompt on the haiku dataset."""

from __future__ import annotations

__all__ = [
    "create_judge_graph",
    "prepare_results",
    "run_experiment",
    "run_inference",
    "run_inference_pipeline",
]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
from coola.utils.timing import timeblock
from langchain.chat_models import init_chat_model
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from argos.nodes import HaikuJudgeState, make_haiku_judge_node
from argos.tasks.autoprompt.analysis import (
    analyze_errors,
)
from argos.tasks.autoprompt.dataset import prepare_dataset
from argos.tasks.autoprompt.evaluation import evaluate_metrics
from argos.utils.batching import batchify
from argos.utils.dataframe import concat_and_merge

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel

    from argos.metrics import BinaryClassificationResults
    from argos.tasks.autoprompt.config import ExperimentConfig

logger: logging.Logger = logging.getLogger(__name__)


def create_judge_graph(model: str, system_prompt: str) -> CompiledStateGraph:
    r"""Create the graph of the haiku judge.

    Args:
        model: The model of the haiku judge.
        system_prompt: The prompt of the judge.

    Returns:
        The graph of the haiku judge.
    """
    logger.info("Creating graph...")
    llm: BaseChatModel = init_chat_model(model=model, temperature=0, max_retries=9999)
    model_version = getattr(llm, "model", getattr(llm, "model_name", "Unknown"))
    logger.info(
        f"class: {type(llm).__name__} | model: {model_version} | temperature: {llm.temperature}"
    )

    graph_builder = StateGraph(HaikuJudgeState)

    graph_builder.add_node("judge", make_haiku_judge_node(llm, system_prompt=system_prompt))

    graph_builder.add_edge(START, "judge")
    graph_builder.add_edge("judge", END)

    return graph_builder.compile()


def prepare_results(dataset: pl.DataFrame, outputs: list[dict[Any, Any]]) -> pl.DataFrame:
    r"""Prepare results of haiku generator-judge.

    Args:
        dataset: The dataset of haiku examples.
        outputs: The results of the haiku generator-judge.

    Returns:
        The results of the haiku generator-judge in a DataFrame.
    """
    cols = [
        "topic",
        "haiku",
        "score",
        "passed",
        "target",
        "structure_passed",
        "structure_target",
        "topic_passed",
        "topic_target",
        "reasoning",
    ]
    flat_data = [
        {**{k: v for k, v in row.items() if k != "evaluation"}, **row["evaluation"].model_dump()}
        for row in outputs
    ]
    return concat_and_merge(pl.DataFrame(flat_data), dataset).select(cols)


def run_experiment(config: ExperimentConfig) -> dict[str, BinaryClassificationResults]:
    r"""Run haiku generator-judge evaluation.

    Args:
        config: The experiment configuration.

    Returns:
        The evaluated metrics.
    """
    path_results = config.path_experiment.joinpath("results.parquet")
    if not path_results.is_file():
        logger.info(f"No results found at {path_results}")
        run_inference(
            model=config.judge_model,
            system_prompt=config.judge_system_prompt,
            path_results=path_results,
        )

    logger.info(f"Reading results from {path_results}")
    results = pl.read_parquet(path_results)
    with pl.Config(tbl_cols=-1, tbl_rows=10):
        logger.info(f"\n{results}")

    analyze_errors(results, path=config.path_experiment)

    return evaluate_metrics(results)


def run_inference(
    model: str, system_prompt: str, path_results: Path, batch_size: int = 20
) -> pl.DataFrame:
    r"""Run inference and store the results in a parquet file.

    Args:
        model: The name of the model to run inference.
        system_prompt: The prompt of the judge.
        path_results: The path of the parquet file to store the results.
        batch_size: The batch size for inference.

    Returns:
        The DataFrame containing the results of the inference.
    """
    graph = create_judge_graph(model=model, system_prompt=system_prompt)
    logger.info(f"\n{graph.get_graph().draw_ascii()}")

    dataset = prepare_dataset()
    results = run_inference_pipeline(dataset=dataset, graph=graph, batch_size=batch_size)

    logger.info(f"Writing results ({results.shape}) in {path_results}")
    path_results.parent.mkdir(parents=True, exist_ok=True)
    results.write_parquet(path_results)
    return results


def run_inference_pipeline(
    dataset: pl.DataFrame, graph: CompiledStateGraph, batch_size: int = 20
) -> pl.DataFrame:
    r"""Run the inference and returns the results in a DataFrame.

    Args:
        dataset: The dataset to run inference on.
        graph: The graph of the haiku judge.
        batch_size: The batch size for inference.

    Returns:
        The results of the inference.
    """
    logger.info(f"Running inference with {batch_size:,} batches...")
    outputs = []
    examples = list(dataset.iter_rows(named=True))
    batches = batchify(examples, size=batch_size)

    with timeblock(message="LLM inference time: {time}"):
        for index, batch in enumerate(batches):
            logger.info(f"--- Processing Batch {index + 1} ---")
            output = graph.batch(batch, config={"max_concurrency": batch_size})
            outputs.extend(output)

    logger.info("Preparing results...")
    return prepare_results(dataset, outputs)
