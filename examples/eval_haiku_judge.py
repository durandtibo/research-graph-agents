r"""Define a script to test the performance of the haiku judge."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from coola.utils.timing import timeblock
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from argos.datasets import generate_haiku_dataset
from argos.metrics import compute_binary_classification_metrics
from argos.nodes import HaikuJudgeState, make_haiku_judge_node
from argos.utils.batching import batchify
from argos.utils.dataframe import concat_and_merge, summarize_boolean_columns
from argos.utils.logging import configure_logging

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class State(HaikuJudgeState):
    r"""Define the state of the haiku generator-judge system."""


def create_graph(model: str = "ollama:gemma3:4b") -> CompiledStateGraph:
    r"""Create the graph of the haiku generator-judge.

    Returns:
        The graph of the haiku generator-judge.
    """
    llm: BaseChatModel = init_chat_model(model=model, temperature=0)
    model_version = getattr(llm, "model", getattr(llm, "model_name", "Unknown"))
    logger.info(
        f"class: {type(llm).__name__} | model: {model_version} | temperature: {llm.temperature}"
    )

    graph_builder = StateGraph(State)

    graph_builder.add_node("judge", make_haiku_judge_node(llm))

    graph_builder.add_edge(START, "judge")
    graph_builder.add_edge("judge", END)

    # Compile the graph into a runnable app
    return graph_builder.compile()


def evaluate_metrics(results: pl.DataFrame) -> None:
    r"""Evaluate the metrics of the haiku generator-judge.

    Args:
        results: The results of the haiku generator-judge.
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


def run_inference(model: str, path_results: Path) -> pl.DataFrame:
    r"""Run inference and store the results in a parquet file.

    Args:
        model: The name of the model to run inference.
        path_results: The path of the parquet file to store the results.

    Returns:
        The DataFrame containing the results of the inference.
    """
    graph = create_graph(model=model)
    logger.info(f"\n{graph.get_graph().draw_ascii()}")

    dataset = prepare_dataset()

    outputs = []
    examples = list(dataset.iter_rows(named=True))
    batch_size = 20
    with timeblock(message="LLM inference time: {time}"):
        for index, batch in enumerate(batchify(examples, size=batch_size)):
            logger.info(f"--- Processing Batch {index + 1} ---")
            outputs.extend(graph.batch(batch, config={"max_concurrency": batch_size}))

    results = prepare_results(dataset, outputs)
    logger.info(f"Writing results ({results.shape}) in {path_results}")
    results.write_parquet(path_results)
    return results


def main() -> None:
    r"""Define the main function to test the haiku judge system."""
    model = "ollama:smollm:135m"
    # model = "ollama:gemma3:1b"
    # model = "anthropic:claude-haiku-4-5-20251001"

    path_results = (
        Path(__file__)
        .resolve()
        .parent.parent.joinpath("results")
        .joinpath("haiku_judge")
        .joinpath(model.replace(":", "_"))
        .joinpath("results.parquet")
    )
    path_results.parent.mkdir(parents=True, exist_ok=True)

    if not path_results.is_file():
        run_inference(model=model, path_results=path_results)

    logger.info(f"Reading results from {path_results}")
    results = pl.read_parquet(path_results)
    with pl.Config(tbl_cols=-1, tbl_rows=10):
        logger.info(f"\n{results}")

    # for row in results.iter_rows(named=True):
    #     if row["score"] < 7:
    #         logger.info(f"\n{row}")

    evaluate_metrics(results)


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    load_dotenv()

    main()
