from __future__ import annotations

from typing import Any

import pytest

from argos.meta_agent.analyses import Analysis, AnalysisDict

##################################
#     Tests for AnalysisDict     #
##################################


def test_analysis_dict_repr_empty() -> None:
    assert repr(AnalysisDict({})) == "AnalysisDict()"


def test_analysis_dict_repr_single() -> None:
    assert repr(AnalysisDict({"style": Analysis("style analysis")})) == (
        "AnalysisDict(\n  (style): Analysis(content='style analysis', metadata=None)\n)"
    )


def test_analysis_dict_repr_multiple() -> None:
    assert repr(
        AnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
            }
        )
    ) == (
        "AnalysisDict(\n"
        "  (style): Analysis(content='style analysis', metadata=None)\n"
        "  (semantic): Analysis(content='semantic analysis', metadata=None)\n"
        ")"
    )


def test_analysis_dict_str_empty() -> None:
    assert str(AnalysisDict({})) == "AnalysisDict()"


def test_analysis_dict_str_single() -> None:
    assert str(AnalysisDict({"style": Analysis("style analysis")})) == (
        "AnalysisDict(\n"
        "  (style): Analysis(\n"
        "      (content): style analysis\n"
        "      (metadata): None\n"
        "    )\n"
        ")"
    )


def test_analysis_dict_str_multiple() -> None:
    assert str(
        AnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
            }
        )
    ) == (
        "AnalysisDict(\n"
        "  (style): Analysis(\n"
        "      (content): style analysis\n"
        "      (metadata): None\n"
        "    )\n"
        "  (semantic): Analysis(\n"
        "      (content): semantic analysis\n"
        "      (metadata): None\n"
        "    )\n"
        ")"
    )


def test_analysis_dict_equal_true() -> None:
    assert AnalysisDict({"style": Analysis("style analysis")}).equal(
        AnalysisDict({"style": Analysis("style analysis")})
    )


def test_analysis_dict_equal_true_empty() -> None:
    assert AnalysisDict({}).equal(AnalysisDict({}))


def test_analysis_dict_equal_false_different_keys() -> None:
    assert not AnalysisDict({"style": Analysis("my analysis")}).equal(
        AnalysisDict({"semantic": Analysis("my analysis")})
    )


def test_analysis_dict_equal_false_different_values() -> None:
    assert not AnalysisDict({"style": Analysis("style analysis")}).equal(
        AnalysisDict({"style": Analysis("other analysis")})
    )


def test_analysis_dict_equal_false_different_number_of_analyses() -> None:
    assert not AnalysisDict({"style": Analysis("style analysis")}).equal(
        AnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
            }
        )
    )


def test_analysis_dict_equal_false_different_type() -> None:
    assert not AnalysisDict({}).equal({})


def test_analysis_dict_equal_false_different_type_child() -> None:
    class Child(AnalysisDict): ...

    assert not AnalysisDict({}).equal(Child({}))


def test_analysis_dict_equal_nan_false_by_default() -> None:
    assert not AnalysisDict({"style": Analysis(float("nan"))}).equal(
        AnalysisDict({"style": Analysis(float("nan"))})
    )


def test_analysis_dict_equal_nan_true() -> None:
    assert AnalysisDict({"style": Analysis(float("nan"))}).equal(
        AnalysisDict({"style": Analysis(float("nan"))}), equal_nan=True
    )


@pytest.mark.parametrize(
    ("analyses", "expected"),
    [
        pytest.param({}, {}, id="empty"),
        pytest.param(
            {"style": Analysis("style analysis")},
            {"style": "style analysis"},
            id="single_string",
        ),
        pytest.param(
            {"style": Analysis("style analysis"), "semantic": Analysis("semantic analysis")},
            {"style": "style analysis", "semantic": "semantic analysis"},
            id="multiple_strings",
        ),
        pytest.param(
            {"score": Analysis(0.9)},
            {"score": 0.9},
            id="float_value",
        ),
        pytest.param(
            {"metrics": Analysis({"loss": 0.5, "accuracy": 0.9})},
            {"metrics": {"loss": 0.5, "accuracy": 0.9}},
            id="nested_dict",
        ),
        pytest.param(
            {"scores": Analysis([0.5, 0.9])},
            {"scores": [0.5, 0.9]},
            id="list_value",
        ),
        pytest.param(
            {"style": AnalysisDict({"sub": Analysis("sub analysis")})},
            {"style": {"sub": "sub analysis"}},
            id="nested_analysis_dict",
        ),
    ],
)
def test_analysis_dict_to_primitive(analyses: Any, expected: Any) -> None:
    assert AnalysisDict(analyses).to_primitive() == expected
