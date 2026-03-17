from __future__ import annotations

import argos


def test_version_exists() -> None:
    assert hasattr(argos, "__version__")
    assert isinstance(argos.__version__, str)
