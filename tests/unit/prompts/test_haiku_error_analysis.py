from __future__ import annotations

from argos.prompts.haiku_error_analysis import HAIKU_ERROR_ANALYSIS_SYSTEM_PROMPT_1


def test_haiku_error_analysis_system_prompt_1_is_string() -> None:
    assert isinstance(HAIKU_ERROR_ANALYSIS_SYSTEM_PROMPT_1, str)


def test_haiku_error_analysis_system_prompt_1_is_not_empty() -> None:
    assert HAIKU_ERROR_ANALYSIS_SYSTEM_PROMPT_1.strip()
