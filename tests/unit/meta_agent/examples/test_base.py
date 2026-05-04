from __future__ import annotations

from coola.equality.tester import get_default_registry

from argos.meta_agent.examples import BaseExample


def test_has_equality_tester_for_base_example() -> None:
    assert get_default_registry().has_equality_tester(BaseExample)
