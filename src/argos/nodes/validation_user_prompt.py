from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from argos.states.user_prompt import UserPromptState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


logger = logging.getLogger(__name__)


class PromptEvaluation(BaseModel):
    """Schema for evaluating if a user prompt matches a target topic."""

    topic_match: str = Field(
        description=(
            "Must be exactly 'Yes', 'No', or 'Partial'. "
            "Does the prompt primarily discuss the Target Topic?"
        )
    )
    identified_topic: str = Field(
        description="State the actual main topic of the user's prompt in 1-5 words."
    )
    user_intent: str = Field(
        description=(
            "Summarize exactly what the user is trying to achieve, ask, or convey in 1-2 sentences."
        )
    )
    clarity_rating: str = Field(
        description=(
            "Must be exactly 'High', 'Medium', or 'Low'. How clear and coherent is the request?"
        )
    )
    evaluation_notes: str = Field(
        description="Briefly explain your reasoning for the Topic Match and what might be missing."
    )


# TODO: can a human understand the prompt?

# SYSTEM_PROMPT = """# SYSTEM PROMPT
#
# **Role:** You are an expert AI Content Evaluator and Comprehension Checker.
#
# **Task:** Your objective is to analyze the user's prompt, verify if it aligns with the designated target topic, and demonstrate a clear understanding of the user's underlying intent. Do not answer the user's prompt directly; your only job is to evaluate it.
#
# **Target Topic:** Python programming
#
# **Instructions:** Carefully read the user's input. Evaluate the text and output your analysis using the exact structure below. Be objective and concise.
#
# **Output Format:**
# Please format your response exactly as follows:
#
# * **Topic Match:** [Yes / No / Partial] - Does the prompt primarily discuss the Target Topic?
# * **Identified Topic:** [State the actual main topic of the user's prompt in 1-5 words.]
# * **User Intent:** [Summarize exactly what the user is trying to achieve, ask, or convey in 1-2 sentences. This proves you understand the prompt.]
# * **Clarity Rating:** [High / Medium / Low] - How clear and coherent is the user's request?
# * **Evaluation Notes:** [Briefly explain your reasoning for the Topic Match and provide any context on what might be missing from the prompt.]"""


system_prompt = """
You are an expert AI Content Evaluator and Comprehension Checker.
Your objective is to analyze the user's prompt, verify if it aligns with the
designated target topic, and demonstrate a clear understanding of their intent.
Do not answer the user's prompt directly; your only job is to evaluate it.

**Target Topic:** Python programming
"""


class UserPromptValidator(Callable[[UserPromptState], dict]):
    def __init__(self, chat_model: BaseChatModel) -> None:
        self._chat_model = chat_model.with_structured_output(PromptEvaluation)

    def __call__(self, state: UserPromptState) -> dict:
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "User Prompt: {user_prompt}")]
        )

        evaluation_chain = prompt | self._chat_model

        result = evaluation_chain.invoke({"user_prompt": state["user_prompt"]})
        logger.info(f"\n{result}")
        return {"evaluation_result": result}
