from __future__ import annotations

import pytest

from argos.utils.prompt import check_non_empty_prompt

EMPTY_PROMPTS = ["", " ", "\n\n"]

############################################
#     Tests for check_non_empty_prompt     #
############################################


@pytest.mark.parametrize("prompt", ["a", " meow ", "meow\nmeow\nmeow"])
def test_check_non_empty_prompt_non_empty(prompt: str) -> None:
    check_non_empty_prompt(prompt)


@pytest.mark.parametrize("prompt", EMPTY_PROMPTS)
def test_check_non_empty_prompt_empty(prompt: str) -> None:
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        check_non_empty_prompt(prompt)
