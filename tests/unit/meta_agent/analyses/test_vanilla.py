from __future__ import annotations

from argos.meta_agent.analyses import Analysis

##############################
#     Tests for Analysis     #
##############################


def test_analysis_repr_empty() -> None:
    assert repr(Analysis("")) == "Analysis(content_len=0)"


def test_analysis_repr() -> None:
    assert repr(Analysis("my analysis")) == "Analysis(content_len=11)"


def test_analysis_str_empty() -> None:
    assert str(Analysis("")) == "Analysis(content_len=0)"


def test_analysis_str() -> None:
    assert str(Analysis("my analysis")) == "Analysis(content_len=11)"


def test_analysis_equal_true() -> None:
    assert Analysis("my analysis").equal(Analysis("my analysis"))


def test_analysis_equal_true_empty() -> None:
    assert Analysis("").equal(Analysis(""))


def test_analysis_equal_false_different_content() -> None:
    assert not Analysis("my analysis").equal(Analysis("other analysis"))


def test_analysis_equal_false_different_type() -> None:
    assert not Analysis("my analysis").equal("my analysis")


def test_analysis_equal_false_different_type_child() -> None:
    class Child(Analysis): ...

    assert not Analysis("my analysis").equal(Child("my analysis"))


def test_analysis_to_dict() -> None:
    assert Analysis("my analysis").to_dict() == {"content": "my analysis"}


def test_analysis_to_dict_empty() -> None:
    assert Analysis("").to_dict() == {"content": ""}


def test_analysis_to_markdown() -> None:
    assert Analysis("my analysis").to_markdown() == "my analysis"


def test_analysis_to_markdown_empty() -> None:
    assert Analysis("").to_markdown() == "_Empty analysis_"
