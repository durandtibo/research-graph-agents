r"""Implement a example about a haiku generator-judge system."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coola.utils.timing import timeblock
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from argos.nodes import (
    HaikuJudgeState,
    make_haiku_generator_node,
    make_haiku_judge_node,
)
from argos.utils.logging import configure_logging

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class State(HaikuJudgeState):
    r"""Define the state of the haiku generator-judge system."""


def create_graph() -> CompiledStateGraph:
    r"""Create the graph of the haiku generator-judge.

    Returns:
        The graph of the haiku generator-judge.
    """
    llm: BaseChatModel = ChatOllama(model="gemma3:1b")
    logger.info(f"LLM model={llm.model}")
    llm_judge: BaseChatModel = llm  # ChatOllama(model="gemma3:4b", temperature=0)
    logger.info(f"LLM judge model={llm_judge.model}")

    graph_builder = StateGraph(State)

    graph_builder.add_node("poet", make_haiku_generator_node(llm))
    graph_builder.add_node("judge", make_haiku_judge_node(llm_judge))

    graph_builder.add_edge(START, "poet")
    graph_builder.add_edge("poet", "judge")
    graph_builder.add_edge("judge", END)

    # Compile the graph into a runnable app
    return graph_builder.compile()


def main() -> None:
    r"""Define the main function of the program."""
    graph = create_graph()
    logger.info(f"\n{graph.get_graph().draw_ascii()}")

    topics = ["cats", "rain", "the ocean"]
    for topic in topics:
        with timeblock(message="LLM inference time: {time}"):
            result = graph.invoke({"topic": topic})
        logger.info(f"\n[{topic}]:\n{result['haiku']}")
        logger.info(f"judgement\n{result['evaluation']}")


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    load_dotenv()

    main()
