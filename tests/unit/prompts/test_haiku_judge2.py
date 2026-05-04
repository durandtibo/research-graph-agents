from __future__ import annotations

import pytest

from argos.prompts.haiku_judge2 import (
    HAIKU_JUDGE_SYSTEM_PROMPT_0,
    HAIKU_JUDGE_SYSTEM_PROMPT_NO_REASONING_0,
    HAIKU_JUDGE_SYSTEM_PROMPT_NO_REASONING_1,
)


@pytest.mark.parametrize(
    "system_prompt",
    [
        HAIKU_JUDGE_SYSTEM_PROMPT_0,
        HAIKU_JUDGE_SYSTEM_PROMPT_NO_REASONING_0,
        HAIKU_JUDGE_SYSTEM_PROMPT_NO_REASONING_1,
    ],
)
def test_system_prompt(system_prompt: str) -> None:
    assert isinstance(system_prompt, str)


@pytest.mark.parametrize(
    "system_prompt",
    [
        HAIKU_JUDGE_SYSTEM_PROMPT_0,
        HAIKU_JUDGE_SYSTEM_PROMPT_NO_REASONING_0,
        HAIKU_JUDGE_SYSTEM_PROMPT_NO_REASONING_1,
    ],
)
def test_system_prompt_is_not_empty(system_prompt: str) -> None:
    assert system_prompt.strip()
