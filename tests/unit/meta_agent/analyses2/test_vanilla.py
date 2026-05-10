from __future__ import annotations

from typing import Any

import pytest

from argos.meta_agent.analyses2 import Analysis

##############################
#     Tests for Analysis     #
##############################


def test_analysis_repr_short_content() -> None:
    assert repr(Analysis("my analysis")) == "Analysis(content='my analysis', metadata=None)"


def test_analysis_repr_long_content() -> None:
    content = "a" * 100
    assert repr(Analysis(content)) == f"Analysis(content='{'a' * 47}...', metadata=None)"


def test_analysis_repr_with_metadata() -> None:
    assert repr(Analysis("my analysis", metadata={"source": "math"})) == (
        "Analysis(content='my analysis', metadata={'source': 'math'})"
    )


def test_analysis_str() -> None:
    assert str(Analysis("my analysis")) == (
        "Analysis(\n  (content): my analysis\n  (metadata): None\n)"
    )


def test_analysis_str_with_metadata() -> None:
    assert str(Analysis("my analysis", metadata={"source": "math"})) == (
        "Analysis(\n  (content): my analysis\n  (metadata): {'source': 'math'}\n)"
    )


def test_analysis_equal_true() -> None:
    assert Analysis("my analysis").equal(Analysis("my analysis"))


def test_analysis_equal_true_with_metadata() -> None:
    assert Analysis("my analysis", metadata={"source": "math"}).equal(
        Analysis("my analysis", metadata={"source": "math"})
    )


def test_analysis_equal_true_empty() -> None:
    assert Analysis("").equal(Analysis(""))


def test_analysis_equal_false_different_content() -> None:
    assert not Analysis("my analysis").equal(Analysis("other analysis"))


def test_analysis_equal_false_different_metadata() -> None:
    assert not Analysis("my analysis", metadata={"source": "math"}).equal(
        Analysis("my analysis", metadata={"source": "science"})
    )


def test_analysis_equal_false_metadata_vs_none() -> None:
    assert not Analysis("my analysis", metadata={"source": "math"}).equal(Analysis("my analysis"))


def test_analysis_equal_false_different_type() -> None:
    assert not Analysis("my analysis").equal("my analysis")


def test_analysis_equal_false_different_type_child() -> None:
    class Child(Analysis): ...

    assert not Analysis("my analysis").equal(Child("my analysis"))


def test_analysis_equal_nan_false_by_default() -> None:
    assert not Analysis(float("nan")).equal(Analysis(float("nan")))


def test_analysis_equal_nan_true() -> None:
    assert Analysis(float("nan")).equal(Analysis(float("nan")), equal_nan=True)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("my analysis", id="string"),
        pytest.param("", id="empty_string"),
        pytest.param(3.14, id="float"),
        pytest.param(42, id="integer"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="none"),
        pytest.param({"key": "value"}, id="dict"),
        pytest.param({}, id="empty_dict"),
        pytest.param({"nested": {"key": "value"}}, id="nested_dict"),
        pytest.param([1, 2, 3], id="list"),
        pytest.param([], id="empty_list"),
        pytest.param([[1, 2], [3, 4]], id="nested_list"),
        pytest.param((1, 2, 3), id="tuple"),
        pytest.param((), id="empty_tuple"),
        pytest.param([{"key": "value"}], id="list_of_dicts"),
    ],
)
def test_analysis_to_primitive(content: Any) -> None:
    assert Analysis(content).to_primitive() == content


def test_analysis_to_primitive_returns_same_object() -> None:
    content = "my analysis"
    assert Analysis(content).to_primitive() is content


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("my analysis", '"my analysis"', id="string"),
        pytest.param("", '""', id="empty_string"),
        pytest.param(3.14, "3.14", id="float"),
        pytest.param(42, "42", id="integer"),
        pytest.param(True, "true", id="boolean"),
        pytest.param(None, "null", id="none"),
        pytest.param({"key": "value"}, '{"key": "value"}', id="dict"),
        pytest.param({}, "{}", id="empty_dict"),
        pytest.param(
            {"nested": {"key": "value"}}, '{"nested": {"key": "value"}}', id="nested_dict"
        ),
        pytest.param([1, 2, 3], "[1, 2, 3]", id="list"),
        pytest.param([], "[]", id="empty_list"),
        pytest.param([[1, 2], [3, 4]], "[[1, 2], [3, 4]]", id="nested_list"),
        pytest.param([{"key": "value"}], '[{"key": "value"}]', id="list_of_dicts"),
    ],
)
def test_analysis_to_json(content: Any, expected: Any) -> None:
    assert Analysis(content).to_json() == expected


def test_analysis_to_json_indent() -> None:
    assert Analysis({"key": "value"}).to_json(indent=2) == '{\n  "key": "value"\n}'


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("my analysis", "my analysis\n...\n", id="string"),
        pytest.param("", "''\n", id="empty_string"),
        pytest.param(3.14, "3.14\n...\n", id="float"),
        pytest.param(42, "42\n...\n", id="integer"),
        pytest.param(True, "true\n...\n", id="boolean"),
        pytest.param(None, "null\n...\n", id="none"),
        pytest.param({"key": "value"}, "key: value\n", id="dict"),
        pytest.param({}, "{}\n", id="empty_dict"),
        pytest.param({"nested": {"key": "value"}}, "nested:\n  key: value\n", id="nested_dict"),
        pytest.param([1, 2, 3], "- 1\n- 2\n- 3\n", id="list"),
        pytest.param([], "[]\n", id="empty_list"),
        pytest.param([[1, 2], [3, 4]], "- - 1\n  - 2\n- - 3\n  - 4\n", id="nested_list"),
        pytest.param([{"key": "value"}], "- key: value\n", id="list_of_dicts"),
    ],
)
def test_analysis_to_yaml(content: Any, expected: Any) -> None:
    assert Analysis(content).to_yaml() == expected


def test_analysis_to_yaml_default_flow_style() -> None:
    assert Analysis({"key": "value"}).to_yaml(default_flow_style=True) == "{key: value}\n"
