from __future__ import annotations

from coola.equality.tester import get_default_registry

from argos.meta_agent.results import BaseResult


def test_has_equality_teste_for_base_result() -> None:
    assert get_default_registry().has_equality_tester(BaseResult)
