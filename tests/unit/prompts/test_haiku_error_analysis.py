from __future__ import annotations

import pytest

from argos.prompts.haiku_error_analysis import HAIKU_ERROR_ANALYSIS_SYSTEM_PROMPT_1


@pytest.mark.parametrize(
    "system_prompt",
    [HAIKU_ERROR_ANALYSIS_SYSTEM_PROMPT_1],
)
def test_system_prompt(system_prompt: str) -> None:
    assert isinstance(system_prompt, str)


def test_system_prompt_is_not_empty() -> None:
    assert HAIKU_ERROR_ANALYSIS_SYSTEM_PROMPT_1.strip()
