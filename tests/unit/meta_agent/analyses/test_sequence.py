from __future__ import annotations

from typing import Any

import pytest

from argos.meta_agent.analyses import Analysis, AnalysisDict, AnalysisList

##################################
#     Tests for AnalysisList     #
##################################


def test_analysis_list_repr_empty() -> None:
    assert repr(AnalysisList([])) == "AnalysisList()"


def test_analysis_list_repr_single() -> None:
    assert repr(AnalysisList([Analysis("style analysis")])) == (
        "AnalysisList(\n  (0): Analysis(content='style analysis', metadata=None)\n)"
    )


def test_analysis_list_repr_multiple() -> None:
    assert repr(
        AnalysisList(
            [
                Analysis("style analysis"),
                Analysis("semantic analysis"),
            ]
        )
    ) == (
        "AnalysisList(\n"
        "  (0): Analysis(content='style analysis', metadata=None)\n"
        "  (1): Analysis(content='semantic analysis', metadata=None)\n"
        ")"
    )


def test_analysis_list_str_empty() -> None:
    assert str(AnalysisList([])) == "AnalysisList()"


def test_analysis_list_str_single() -> None:
    assert str(AnalysisList([Analysis("style analysis")])) == (
        "AnalysisList(\n"
        "  (0): Analysis(\n"
        "      (content): style analysis\n"
        "      (metadata): None\n"
        "    )\n"
        ")"
    )


def test_analysis_list_str_multiple() -> None:
    assert str(
        AnalysisList(
            [
                Analysis("style analysis"),
                Analysis("semantic analysis"),
            ]
        )
    ) == (
        "AnalysisList(\n"
        "  (0): Analysis(\n"
        "      (content): style analysis\n"
        "      (metadata): None\n"
        "    )\n"
        "  (1): Analysis(\n"
        "      (content): semantic analysis\n"
        "      (metadata): None\n"
        "    )\n"
        ")"
    )


def test_analysis_list_equal_true() -> None:
    assert AnalysisList([Analysis("style analysis")]).equal(
        AnalysisList([Analysis("style analysis")])
    )


def test_analysis_list_equal_true_empty() -> None:
    assert AnalysisList([]).equal(AnalysisList([]))


def test_analysis_list_equal_false_different_values() -> None:
    assert not AnalysisList([Analysis("style analysis")]).equal(
        AnalysisList([Analysis("other analysis")])
    )


def test_analysis_list_equal_false_different_length() -> None:
    assert not AnalysisList([Analysis("style analysis")]).equal(
        AnalysisList([Analysis("style analysis"), Analysis("semantic analysis")])
    )


def test_analysis_list_equal_false_different_order() -> None:
    assert not AnalysisList([Analysis("style analysis"), Analysis("semantic analysis")]).equal(
        AnalysisList([Analysis("semantic analysis"), Analysis("style analysis")])
    )


def test_analysis_list_equal_false_different_type() -> None:
    assert not AnalysisList([]).equal([])


def test_analysis_list_equal_false_different_type_child() -> None:
    class Child(AnalysisList): ...

    assert not AnalysisList([]).equal(Child([]))


def test_analysis_list_equal_nan_false_by_default() -> None:
    assert not AnalysisList([Analysis(float("nan"))]).equal(AnalysisList([Analysis(float("nan"))]))


def test_analysis_list_equal_nan_true() -> None:
    assert AnalysisList([Analysis(float("nan"))]).equal(
        AnalysisList([Analysis(float("nan"))]), equal_nan=True
    )


@pytest.mark.parametrize(
    ("analyses", "expected"),
    [
        pytest.param([], [], id="empty"),
        pytest.param(
            [Analysis("style analysis")],
            ["style analysis"],
            id="single_string",
        ),
        pytest.param(
            [Analysis("style analysis"), Analysis("semantic analysis")],
            ["style analysis", "semantic analysis"],
            id="multiple_strings",
        ),
        pytest.param(
            [Analysis(0.9), Analysis(0.5)],
            [0.9, 0.5],
            id="float_values",
        ),
        pytest.param(
            [Analysis({"loss": 0.5, "accuracy": 0.9})],
            [{"loss": 0.5, "accuracy": 0.9}],
            id="dict_value",
        ),
        pytest.param(
            [Analysis([1, 2, 3])],
            [[1, 2, 3]],
            id="list_value",
        ),
        pytest.param(
            [AnalysisList([Analysis("sub analysis")])],
            [["sub analysis"]],
            id="nested_analysis_list",
        ),
        pytest.param(
            [AnalysisDict({"key": Analysis("value")})],
            [{"key": "value"}],
            id="nested_analysis_dict",
        ),
    ],
)
def test_analysis_list_to_primitive(analyses: Any, expected: Any) -> None:
    assert AnalysisList(analyses).to_primitive() == expected


@pytest.mark.parametrize(
    ("analyses", "expected"),
    [
        pytest.param([], "[]", id="empty"),
        pytest.param(
            [Analysis("style analysis")],
            '["style analysis"]',
            id="single_string",
        ),
        pytest.param(
            [Analysis("style analysis"), Analysis("semantic analysis")],
            '["style analysis", "semantic analysis"]',
            id="multiple_strings",
        ),
        pytest.param(
            [Analysis(0.9)],
            "[0.9]",
            id="float_value",
        ),
        pytest.param(
            [AnalysisDict({"key": Analysis("value")})],
            '[{"key": "value"}]',
            id="nested_analysis_dict",
        ),
    ],
)
def test_analysis_list_to_json(analyses: Any, expected: str) -> None:
    assert AnalysisList(analyses).to_json() == expected


def test_analysis_list_to_json_indent() -> None:
    analysis = AnalysisList([Analysis("a"), Analysis("b")])
    assert analysis.to_json(indent=2) == '[\n  "a",\n  "b"\n]'


@pytest.mark.parametrize(
    ("analyses", "expected"),
    [
        pytest.param([], "[]\n", id="empty"),
        pytest.param(
            [Analysis("style analysis")],
            "- style analysis\n",
            id="single_string",
        ),
        pytest.param(
            [Analysis("style analysis"), Analysis("semantic analysis")],
            "- style analysis\n- semantic analysis\n",
            id="multiple_strings",
        ),
        pytest.param(
            [Analysis(0.9)],
            "- 0.9\n",
            id="float_value",
        ),
        pytest.param(
            [AnalysisDict({"key": Analysis("value")})],
            "- key: value\n",
            id="nested_analysis_dict",
        ),
    ],
)
def test_analysis_list_to_yaml(analyses: Any, expected: str) -> None:
    assert AnalysisList(analyses).to_yaml() == expected
