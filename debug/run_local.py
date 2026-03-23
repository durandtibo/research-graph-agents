from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from coola.utils.timing import timeblock
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph

from argos.utils.logging import configure_logging
from argos.utils.today import get_today_date

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class State(MessagesState):
    today: str


class LlmCaller(Callable[[State], State]):
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: State) -> State:
        logger.info(f"state={state}")
        logger.info(f"Call LLM with:\n{state['messages']}")
        # The reducer handles the history; we just pass the full list to the LLM
        response = self.llm.invoke(state["messages"])
        logger.info(f"usage_metadata={response.usage_metadata}")
        # We only return the NEW message in a list
        return {"messages": [response]}


def create_graph() -> CompiledStateGraph:
    llm: BaseChatModel = ChatOllama(model="gemma3:1b", temperature=0)
    logger.info(f"Model={llm.model}")

    graph_builder = StateGraph(State)

    # Add our single node
    graph_builder.add_node("agent", LlmCaller(llm))

    # Define the flow: START -> agent -> END
    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("agent", END)

    # Compile the graph into a runnable app
    return graph_builder.compile()


def main() -> None:
    # llm: BaseChatModel = ChatOllama(model="llama3.2")
    llm: BaseChatModel = ChatOllama(model="gemma3:1b", temperature=0)
    logger.info(f"Model={llm.model}")
    logger.info(f"\n{llm.get_graph().draw_ascii()}")

    messages = [
        SystemMessage(
            content="You are a helpful assistant. Replace each bear occurrence by a bear emoji."
        ),
        HumanMessage(content="Write a haiku bout grizzly bears"),
    ]
    with timeblock(message="LLM inference time: {time}"):
        response = llm.invoke(messages)
    logger.info(f"response:\n{response.content}")
    logger.info(f"usage_metadata={response.usage_metadata}")


def main2() -> None:
    # llm: BaseChatModel = ChatOllama(model="llama3.2")

    graph = create_graph()
    logger.info(f"\n{graph.get_graph().draw_ascii()}")

    messages = {
        "today": get_today_date(),
        "messages": [
            {"role": "user", "content": "Write a haiku bout grizzly bears"},
            {"role": "user", "content": "What is today's date?"},
            {"role": "user", "content": "What are the top-tier ML conferences?"},
        ],
    }
    with timeblock(message="LLM inference time: {time}"):
        result = graph.invoke(messages)

    for message in result["messages"]:
        logger.info("\n" + message.pretty_repr())


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    load_dotenv()

    main2()
