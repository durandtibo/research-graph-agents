from __future__ import annotations

from argos.meta_agent.analyses import (
    Analysis,
    AnalysisDict,
    BulletPointAnalysisDict,
    YamlAnalysisDict,
)

##################################
#     Tests for AnalysisDict     #
##################################


def test_analysis_dict_repr_empty() -> None:
    assert repr(AnalysisDict({})) == "AnalysisDict(count=0)"


def test_analysis_dict_repr_single() -> None:
    assert repr(AnalysisDict({"style": Analysis("style analysis")})) == "AnalysisDict(count=1)"


def test_analysis_dict_repr_multiple() -> None:
    assert (
        repr(
            AnalysisDict(
                {
                    "style": Analysis("style analysis"),
                    "semantic": Analysis("semantic analysis"),
                }
            )
        )
        == "AnalysisDict(count=2)"
    )


def test_analysis_dict_str_empty() -> None:
    assert str(AnalysisDict({})) == "AnalysisDict()"


def test_analysis_dict_str_single() -> None:
    result = str(AnalysisDict({"style": Analysis("style analysis")}))
    assert result == "AnalysisDict(\n  (style): Analysis(content_len=14, metadata=None)\n)"


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


def test_analysis_dict_equal_nan_false_by_default() -> None:
    assert not AnalysisDict({"style": Analysis(float("nan"))}).equal(
        AnalysisDict({"style": Analysis(float("nan"))})
    )


def test_analysis_dict_equal_nan_true() -> None:
    assert AnalysisDict({"style": Analysis(float("nan"))}).equal(
        AnalysisDict({"style": Analysis(float("nan"))}), equal_nan=True
    )


def test_analysis_dict_from_dict() -> None:
    analyses = {"style": Analysis("style analysis")}
    assert AnalysisDict.from_dict({"analyses": analyses}).equal(AnalysisDict(analyses))


def test_analysis_dict_from_dict_empty() -> None:
    assert AnalysisDict.from_dict({"analyses": {}}).equal(AnalysisDict({}))


def test_analysis_dict_to_dict_empty() -> None:
    assert AnalysisDict({}).to_dict() == {"analyses": {}}


def test_analysis_dict_to_dict_single() -> None:
    assert AnalysisDict({"style": Analysis("style analysis")}).to_dict() == {
        "analyses": {"style": Analysis("style analysis")}
    }


def test_analysis_dict_to_dict_multiple() -> None:
    assert AnalysisDict(
        {
            "style": Analysis("style analysis"),
            "semantic": Analysis("semantic analysis"),
        }
    ).to_dict() == {
        "analyses": {
            "style": Analysis("style analysis"),
            "semantic": Analysis("semantic analysis"),
        }
    }


def test_analysis_dict_to_dict_roundtrip() -> None:
    analysis = AnalysisDict(
        {
            "style": Analysis("style analysis"),
            "semantic": Analysis("semantic analysis"),
        }
    )
    assert AnalysisDict.from_dict(analysis.to_dict()).equal(analysis)


def test_analysis_dict_to_text_empty() -> None:
    assert AnalysisDict({}).to_text() == "{}"


def test_analysis_dict_to_text_single() -> None:
    assert AnalysisDict({"style": Analysis("style analysis")}).to_text() == (
        "{'style': 'style analysis'}"
    )


def test_analysis_dict_to_text_multiple() -> None:
    assert (
        AnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
            }
        ).to_text()
        == "{'style': 'style analysis', 'semantic': 'semantic analysis'}"
    )


def test_analysis_dict_to_text_nested() -> None:
    assert (
        AnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
                "other": AnalysisDict(
                    {
                        "cat": Analysis("I am a cat"),
                        "bear": Analysis("I am a bear"),
                    }
                ),
            }
        ).to_text()
        == "{'style': 'style analysis', 'semantic': 'semantic analysis', "
        "'other': \"{'cat': 'I am a cat', 'bear': 'I am a bear'}\"}"
    )


#############################################
#     Tests for BulletPointAnalysisDict     #
#############################################


def test_bullet_point_analysis_dict_to_text_empty() -> None:
    assert BulletPointAnalysisDict({}).to_text() == ""


def test_bullet_point_analysis_dict_to_text_single() -> None:
    assert BulletPointAnalysisDict({"style": Analysis("style analysis")}).to_text() == (
        "- style: style analysis"
    )


def test_bullet_point_analysis_dict_to_text_multiple() -> None:
    assert (
        BulletPointAnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
            }
        ).to_text()
        == "- style: style analysis\n- semantic: semantic analysis"
    )


def test_bullet_point_analysis_dict_to_text_nested() -> None:
    assert (
        BulletPointAnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
                "other": BulletPointAnalysisDict(
                    {
                        "cat": Analysis("I am a cat"),
                        "bear": Analysis("I am a bear"),
                    }
                ),
            }
        ).to_text()
        == "- style: style analysis\n"
        "- semantic: semantic analysis\n"
        "- other:\n"
        "  - cat: I am a cat\n"
        "  - bear: I am a bear"
    )


######################################
#     Tests for YamlAnalysisDict     #
######################################


def test_yaml_analysis_dict_to_text_empty() -> None:
    assert YamlAnalysisDict({}).to_text() == ""


def test_yaml_analysis_dict_to_text_single() -> None:
    assert YamlAnalysisDict({"style": Analysis("style analysis")}).to_text() == (
        "style: style analysis"
    )


def test_yaml_analysis_dict_to_text_multiple() -> None:
    assert (
        YamlAnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
            }
        ).to_text()
        == "style: style analysis\nsemantic: semantic analysis"
    )


def test_yaml_analysis_dict_to_text_nested() -> None:
    assert (
        YamlAnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
                "other": YamlAnalysisDict(
                    {
                        "cat": Analysis("I am a cat"),
                        "bear": Analysis("I am a bear"),
                    }
                ),
            }
        ).to_text()
        == "style: style analysis\n"
        "semantic: semantic analysis\n"
        "other:\n"
        "  cat: I am a cat\n"
        "  bear: I am a bear"
    )
