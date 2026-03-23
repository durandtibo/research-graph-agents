from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph

from argos.constants import DEV_MODEL
from argos.nodes.validation_user_prompt import PromptEvaluation, UserPromptValidator
from argos.states import UserPromptState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

DEFAULT_MODEL = DEV_MODEL


class State(UserPromptState):
    evaluation_result: PromptEvaluation


def create_graph() -> CompiledStateGraph:
    chat_model: BaseChatModel = init_chat_model(
        DEFAULT_MODEL, temperature=0.5, timeout=10, max_tokens=1000
    )

    # Build workflow
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("user_prompt_evaluation", UserPromptValidator(chat_model))

    # Add edges to connect nodes
    workflow.add_edge(START, "user_prompt_evaluation")
    workflow.add_edge("user_prompt_evaluation", END)

    # Compile
    return workflow.compile()


def main() -> None:
    # llm: BaseChatModel = init_chat_model(
    #     DEFAULT_MODEL, temperature=0.5, timeout=10, max_tokens=1000
    # )
    # logger.info(f"\n{llm.get_graph().draw_ascii()}")
    # logger.info(llm.__class__.__mro__)
    #
    # messages = [
    #     SystemMessage(
    #         content="You are a helpful bear assistant. Start each response with a bear emoji."
    #     ),
    #     HumanMessage(content="Analyze the latest trends in CRISPR."),
    # ]
    # response = llm.invoke(messages)
    # logger.info(f"\n{response.content}")
    #
    # agent: CompiledStateGraph = create_agent(
    #     model=DEFAULT_MODEL,
    #     system_prompt="You are a helpful assistant",
    # )
    # logger.info(f"\n{agent.get_graph().draw_ascii()}")
    # logger.info(agent.__class__.__mro__)

    chain = create_graph()
    logger.info(f"\n{chain.get_graph().draw_ascii()}")

    output = chain.invoke({"user_prompt": "Analyze the latest trends in CRISPR."})
    logger.info(f"\n{output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    main()
