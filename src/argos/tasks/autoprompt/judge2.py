r"""Contain code to run the autoprompt on the haiku dataset."""

from __future__ import annotations

__all__ = ["create_judge_graph"]

import logging
from typing import TYPE_CHECKING

from langchain.chat_models import init_chat_model
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from argos.models.haiku_judge import create_haiku_judge_model
from argos.nodes import HaikuJudgeState, HaikuState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from argos.tasks.autoprompt.config import LlmConfig

logger: logging.Logger = logging.getLogger(__name__)


def create_judge_graph(config: LlmConfig) -> CompiledStateGraph:
    r"""Create the graph of the haiku judge.

    Args:
        config: The LLM config.

    Returns:
        The graph of the haiku judge.
    """
    logger.info("Creating judge graph...")
    if config.temperature > 0:
        logger.warning("It is recommended to set temperature to 0 to have a deterministic judge")
    init_kwargs = config.init_kwargs or {}
    llm: BaseChatModel = init_chat_model(
        model=config.model,
        temperature=config.temperature,
        max_retries=config.max_retries,
        **init_kwargs,
    )
    model_version = getattr(llm, "model", getattr(llm, "model_name", "Unknown"))
    logger.info(
        f"class: {type(llm).__name__} | model: {model_version} | temperature: {llm.temperature}"
    )
    judge = create_haiku_judge_model(llm=llm, system_prompt=config.system_prompt)

    def judge_node(state: HaikuState) -> dict:
        return {"evaluation": judge.invoke({"topic": state["topic"], "haiku": state["haiku"]})}

    graph_builder = StateGraph(HaikuJudgeState)

    graph_builder.add_node("judge", judge_node)

    graph_builder.add_edge(START, "judge")
    graph_builder.add_edge("judge", END)

    return graph_builder.compile()
