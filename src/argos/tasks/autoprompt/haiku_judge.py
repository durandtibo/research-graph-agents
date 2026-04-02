r"""Contain code to run the autoprompt on the haiku dataset."""

from __future__ import annotations

__all__ = [
    "create_graph",
    "evaluate_metrics",
    "prepare_dataset",
    "prepare_results",
    "run_inference",
]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
from coola.utils.timing import timeblock
from langchain.chat_models import init_chat_model
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from argos.datasets import generate_haiku_dataset
from argos.metrics import (
    BinaryClassificationResults,
    compute_binary_classification_metrics,
)
from argos.nodes import HaikuJudgeState, make_haiku_judge_node
from argos.utils.batching import batchify
from argos.utils.dataframe import concat_and_merge, summarize_boolean_columns

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger: logging.Logger = logging.getLogger(__name__)


def create_graph(model: str, judge_system_prompt: str) -> CompiledStateGraph:
    r"""Create the graph of the haiku generator-judge.

    Args:
        model: The model of the haiku generator-judge.
        judge_system_prompt: The prompt of the judge-system-prompt.

    Returns:
        The graph of the haiku judge.
    """
    llm: BaseChatModel = init_chat_model(model=model, temperature=0, max_retries=9999)
    model_version = getattr(llm, "model", getattr(llm, "model_name", "Unknown"))
    logger.info(
        f"class: {type(llm).__name__} | model: {model_version} | temperature: {llm.temperature}"
    )

    graph_builder = StateGraph(HaikuJudgeState)

    graph_builder.add_node("judge", make_haiku_judge_node(llm, system_prompt=judge_system_prompt))

    graph_builder.add_edge(START, "judge")
    graph_builder.add_edge("judge", END)

    return graph_builder.compile()


def evaluate_metrics(results: pl.DataFrame) -> dict[str, BinaryClassificationResults]:
    r"""Evaluate the metrics of the haiku generator-judge.

    Args:
        results: The results of the haiku generator-judge.

    Returns:
        The evaluated metrics.
    """
    logger.info(
        f"\n{summarize_boolean_columns(results.select(['target', 'structure_target', 'topic_target']))}"
    )

    overall = compute_binary_classification_metrics(
        results, target_col="target", predict_col="passed"
    )
    logger.info(f"overall\n{overall.to_str()}")

    structure = compute_binary_classification_metrics(
        results, target_col="structure_target", predict_col="structure_passed"
    )
    logger.info(f"structure\n{structure.to_str()}")

    topic = compute_binary_classification_metrics(
        results, target_col="topic_target", predict_col="topic_passed"
    )
    logger.info(f"topic\n{topic.to_str()}")
    return {"overall": overall, "structure": structure, "topic": topic}


def prepare_dataset() -> pl.DataFrame:
    r"""Prepare a dataset of haiku examples.

    Returns:
        A DataFrame containing haiku examples.
    """
    with timeblock(message="Dataset generation time: {time}"):
        dataset = generate_haiku_dataset()

    # uncomment this line to sample a smaller version of the dataset.
    # dataset = dataset.sample(n=5, seed=42)
    with pl.Config(tbl_cols=-1, tbl_rows=10):
        logger.info(f"\n{dataset}")

    stats = summarize_boolean_columns(
        dataset.select(["target", "structure_target", "topic_target"])
    )
    logger.info(f"statistics about the dataset\n{stats}")
    return dataset


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


def run_inference(
    dataset: pl.DataFrame, model_graph: CompiledStateGraph, batch_size: int = 20
) -> pl.DataFrame:
    r"""Run the inference and returns the results in a DataFrame.

    Args:
        dataset: The dataset to run inference on.
        model_graph: The graph of the haiku judge.
        batch_size: The batch size for inference.

    Returns:
        The results of the inference.
    """
    outputs = []
    examples = list(dataset.iter_rows(named=True))
    batches = batchify(examples, size=batch_size)

    with timeblock(message="LLM inference time: {time}"):
        for index, batch in enumerate(batches):
            logger.info(f"--- Processing Batch {index + 1} ---")
            output = model_graph.batch(batch, config={"max_concurrency": batch_size})
            outputs.extend(output)

    logger.info("Preparing results...")
    return prepare_results(dataset, outputs)
