from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coola.utils.timing import timeblock
from dotenv import load_dotenv
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


llm: BaseChatModel = ChatOllama(model="gemma3:1b", temperature=0)


def chatbot_node(state: MessagesState):
    """Passes the conversation history to the LLM and returns the
    response."""
    # The LLM automatically looks at the entire list of messages in the state
    response = llm.invoke(state["messages"])

    # Return the new message. MessagesState will append it to the history.
    return {"messages": [response]}


def create_graph() -> CompiledStateGraph:
    logger.info(f"LLM model={llm.model}")

    graph_builder = StateGraph(State)

    # Add our single node
    graph_builder.add_node("agent", chatbot_node)

    # Define the flow: START -> agent -> END
    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("agent", END)

    # Compile the graph into a runnable app
    return graph_builder.compile()


def main() -> None:
    graph = create_graph()
    logger.info(f"\n{graph.get_graph().draw_ascii()}")

    messages = {
        "today": get_today_date(),
        "messages": [{"role": "user", "content": "Write a haiku bout grizzly bears"}],
    }
    with timeblock(message="LLM inference time: {time}"):
        result = graph.invoke(messages)

    for message in result["messages"]:
        logger.info("\n" + message.pretty_repr())


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    load_dotenv()

    main()
