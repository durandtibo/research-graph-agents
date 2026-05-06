from __future__ import annotations

from argos.meta_agent.analyses import Analysis

##############################
#     Tests for Analysis     #
##############################


def test_analysis_repr_empty() -> None:
    assert repr(Analysis("")) == "Analysis(content_len=0, metadata=None)"


def test_analysis_repr() -> None:
    assert (
        repr(Analysis("my analysis", metadata={"tag": "meow"}))
        == "Analysis(content_len=11, metadata={'tag': 'meow'})"
    )


def test_analysis_str_empty() -> None:
    assert str(Analysis("")) == "Analysis(content_len=0, metadata=None)"


def test_analysis_str() -> None:
    assert (
        str(Analysis("my analysis", metadata={"tag": "meow"}))
        == "Analysis(content_len=11, metadata={'tag': 'meow'})"
    )


def test_analysis_equal_true() -> None:
    assert Analysis("my analysis").equal(Analysis("my analysis"))


def test_analysis_equal_true_empty() -> None:
    assert Analysis("").equal(Analysis(""))


def test_analysis_equal_true_with_metadata() -> None:
    assert Analysis("my analysis", metadata={"tag": "meow"}).equal(
        Analysis("my analysis", metadata={"tag": "meow"})
    )


def test_analysis_equal_false_different_content() -> None:
    assert not Analysis("my analysis").equal(Analysis("other analysis"))


def test_analysis_equal_false_different_metadata() -> None:
    assert not Analysis("my analysis").equal(Analysis("other analysis", metadata={"tag": "meow"}))


def test_analysis_equal_false_different_type() -> None:
    assert not Analysis("my analysis").equal("my analysis")


def test_analysis_equal_false_different_type_child() -> None:
    class Child(Analysis): ...

    assert not Analysis("my analysis").equal(Child("my analysis"))


def test_analysis_from_dict() -> None:
    assert Analysis.from_dict({"content": "my analysis", "metadata": {"tag": "meow"}}).equal(
        Analysis("my analysis", metadata={"tag": "meow"})
    )


def test_analysis_to_dict() -> None:
    assert Analysis("my analysis").to_dict() == {"content": "my analysis", "metadata": None}


def test_analysis_to_dict_empty() -> None:
    assert Analysis("").to_dict() == {"content": "", "metadata": None}


def test_analysis_to_dict_with_metadata() -> None:
    assert Analysis("my analysis", metadata={"tag": "meow"}).to_dict() == {
        "content": "my analysis",
        "metadata": {"tag": "meow"},
    }


def test_analysis_to_text() -> None:
    assert Analysis("my analysis").to_text() == "my analysis"


def test_analysis_to_text_empty() -> None:
    assert Analysis("").to_text() == "_Empty analysis_"
