r"""Implement an example for autoprompt."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from iden.io import save_text
from langchain_core.runnables import RunnableConfig

from argos.autoprompt.haiku.chat_model import create_chat_model
from argos.autoprompt.haiku.config import ExperimentConfig, LlmConfig
from argos.autoprompt.haiku.dataset import prepare_dataset
from argos.autoprompt.haiku.error_analyzer import ErrorAnalyzer
from argos.autoprompt.haiku.error_finder import ErrorFinder
from argos.autoprompt.haiku.evaluator import HaikuJudgeEvaluator
from argos.autoprompt.haiku.inference import InferencePipeline
from argos.autoprompt.haiku.judge import create_judge_graph
from argos.autoprompt.haiku.predictor import Predictor
from argos.autoprompt.haiku.prompt_generator import HistoryPromptGenerator
from argos.models.analysis import create_analyzer_model
from argos.models.prompt_generation import create_prompt_generator_model
from argos.prompts.haiku_error_analysis import HAIKU_ERROR_ANALYSIS_SYSTEM_PROMPT_1
from argos.prompts.haiku_judge2 import (
    HAIKU_JUDGE_SYSTEM_PROMPT_NO_REASONING_1,
)
from argos.prompts.prompt_generation import PROMPT_GENERATOR_SYSTEM_PROMPT_0
from argos.utils.history import BaseHistory, JsonHistory
from argos.utils.logging import configure_logging, log_markdown

if TYPE_CHECKING:
    import polars as pl

logger: logging.Logger = logging.getLogger(__name__)


def generate_next_judge_system_prompt(config: ExperimentConfig, history: BaseHistory) -> str:
    r"""Generate the next judge system prompt.

    Args:
        config: The experiment config.
        history: The history with the previous system prompt.

    Returns:
        The judge system prompt for the iteration.
    """
    if config.iteration == 0:
        logger.info("Using the initial system prompt...")
        return config.judge_system_prompt

    logger.info("Initializing the system prompt generator...")
    prompt_generator = HistoryPromptGenerator(
        history=history.get_values(),
        model=create_prompt_generator_model(
            llm=create_chat_model(config.prompt_generator),
            system_prompt=config.prompt_generator.system_prompt,
        ),
        path=config.path_artifact.joinpath("generated_system_prompt.json"),
    )
    logger.info(f"prompt generator:\n{prompt_generator}")
    return prompt_generator.generate()


def generate_judge_system_prompt(config: ExperimentConfig, history: BaseHistory) -> str:
    r"""Generate the system prompt for the judge.

    Args:
        config: The experiment config.
        history: The history with the previous system prompt.

    Returns:
        The judge system prompt for the iteration.
    """
    logger.info("Generating the judge system prompt...")
    judge_system_prompt = generate_next_judge_system_prompt(config=config, history=history)
    log_markdown(judge_system_prompt, title=f"Judge System Prompt (iteration: {config.iteration})")
    save_text(
        judge_system_prompt, config.path_artifact.joinpath("judge_system_prompt.md"), exist_ok=True
    )
    if config.iteration == 1:
        raise NotImplementedError
    return judge_system_prompt


def generate_predictions(config: ExperimentConfig) -> pl.DataFrame:
    r"""Run the inference pipeline and generate the predictions.

    Args:
        config: The experiment config.
    """
    logger.info("Initializing the inference pipeline...")
    inferpipe = InferencePipeline(
        dataset=prepare_dataset(),
        predictor=Predictor(
            model=create_judge_graph(config.judge),
            batch_size=config.judge.batch_size,
            config=RunnableConfig(max_concurrency=config.batch_size),
        ),
        path=config.path_artifact.joinpath("predictions.parquet"),
    )
    logger.info(f"Inference pipeline:\n{inferpipe}")
    predictions = inferpipe.process()
    logger.info(predictions)
    logger.info(f"columns: {sorted(predictions.columns)}")
    return predictions


def generate_metrics(config: ExperimentConfig, predictions: pl.DataFrame) -> dict[Any, Any]:
    r"""Compute the metrics based on the predictions.

    Args:
        config: The experiment config.
        predictions: The predictions.

    Returns:
        The computed metrics.
    """
    logger.info("Initializing the evaluator...")
    evaluator = HaikuJudgeEvaluator(config.path_artifact.joinpath("metrics.json"))
    logger.info(f"evaluator:\n{evaluator}")
    metrics = evaluator.evaluate(predictions)
    logger.info(f"metrics:\n{metrics}")
    return metrics


def generate_error_analysis(config: ExperimentConfig, predictions: pl.DataFrame) -> str:
    r"""Generate a summary of the error analysis.

    Args:
        config: The experiment config.
        predictions: The predictions.

    Returns:
        The error analysis.
    """
    logger.info("Initializing the error analyzer...")
    system_prompt = config.error_analyzer.system_prompt
    log_markdown(system_prompt, title="Error Analyzer System Prompt")
    analyzer = ErrorAnalyzer(
        error_finder=ErrorFinder(root_path=config.path_artifact),
        model=create_analyzer_model(
            llm=create_chat_model(config.error_analyzer),
            system_prompt=system_prompt,
        ),
        path=config.path_artifact.joinpath("error_analysis.md"),
    )
    logger.info(f"error analyzer:\n{analyzer}")
    analysis = analyzer.analyze(predictions)
    log_markdown(analysis, title=f"Error Analysis (iteration: {config.iteration})")
    return analysis


def run_one_iteration(config: ExperimentConfig, history: BaseHistory) -> None:
    r"""Run one iteration."""
    logger.info("<" * 10 + f" start of iteration {config.iteration} " + ">" * 10)
    state = {}

    config.judge.system_prompt = generate_judge_system_prompt(config=config, history=history)
    state["system_prompt"] = config.judge.system_prompt

    predictions = generate_predictions(config)
    state["metrics"] = generate_metrics(config=config, predictions=predictions)
    state["errors_analysis"] = generate_error_analysis(config=config, predictions=predictions)

    logger.info("<" * 10 + f" end of iteration {config.iteration} " + ">" * 10 + f"\nstate:{state}")
    history.append(state)


def run(config: ExperimentConfig) -> None:
    r"""Run an experiment given a config."""
    logger.info(config)
    history = JsonHistory(config.path_history)
    history.clear()

    for i in range(3):
        config.iteration = i
        run_one_iteration(config=config, history=history)

    logger.info(history.get_values())


def main() -> None:
    r"""Define the main function."""
    path_experiment = (
        Path(__file__)
        .resolve()
        .parent.parent.joinpath("results")
        .joinpath("autoprompt")
        .joinpath("haiku")
    )

    models = [
        # "ollama:smollm:135m",
        # "ollama:gemma3:1b",
        # "ollama:gemma3:4b",
        # "ollama:gemma3:12b",
        "anthropic:claude-haiku-4-5-20251001",
        # "anthropic:claude-sonnet-4-6",
        # "anthropic:claude-opus-4-6",
        # "google_genai:gemini-3.1-flash-lite-preview",
        # "google_genai:gemini-3-flash-preview",
        # "google_genai:gemini-3.1-pro-preview",
        # "openai:gpt-5.4-nano",
        # "openai:gpt-5.4-mini",
        # "openai:gpt-5.4",
    ]
    judge_system_prompts = [
        HAIKU_JUDGE_SYSTEM_PROMPT_NO_REASONING_1,
    ]

    for judge_model in models:
        for judge_system_prompt in judge_system_prompts:
            config = ExperimentConfig(
                judge_model=judge_model,
                judge_system_prompt=judge_system_prompt,
                path_experiment=path_experiment.joinpath(
                    hashlib.sha256(bytes(str(judge_system_prompt), "utf-8")).hexdigest()[:10]
                ).joinpath(judge_model.replace(":", "_")),
                judge=LlmConfig(
                    model=judge_model, system_prompt=judge_system_prompt, batch_size=20
                ),
                error_analyzer=LlmConfig(
                    model="ollama:gemma3:1b",
                    # model="anthropic:claude-haiku-4-5-20251001",
                    system_prompt=HAIKU_ERROR_ANALYSIS_SYSTEM_PROMPT_1,
                ),
                prompt_generator=LlmConfig(
                    model="ollama:gemma3:1b",
                    # model="anthropic:claude-haiku-4-5-20251001",
                    system_prompt=PROMPT_GENERATOR_SYSTEM_PROMPT_0,
                ),
            )
            run(config)


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    load_dotenv()

    main()
