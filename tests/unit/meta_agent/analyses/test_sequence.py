from __future__ import annotations

from argos.meta_agent.analyses import Analysis, AnalysisList, IndentedListAnalysisList

##################################
#     Tests for AnalysisList     #
##################################


def test_analysis_list_repr_empty() -> None:
    assert repr(AnalysisList([])) == "AnalysisList(count=0)"


def test_analysis_list_repr_single() -> None:
    assert repr(AnalysisList([Analysis("style analysis")])) == "AnalysisList(count=1)"


def test_analysis_list_repr_multiple() -> None:
    assert (
        repr(AnalysisList([Analysis("style analysis"), Analysis("semantic analysis")]))
        == "AnalysisList(count=2)"
    )


def test_analysis_list_repr_nested() -> None:
    assert (
        repr(
            AnalysisList(
                [
                    Analysis("style analysis"),
                    Analysis("semantic analysis"),
                    AnalysisList(
                        [
                            Analysis("I am a cat"),
                            Analysis("I am a bear"),
                        ]
                    ),
                ]
            )
        )
        == "AnalysisList(count=3)"
    )


def test_analysis_list_str_empty() -> None:
    assert str(AnalysisList({})) == "AnalysisList()"


def test_analysis_list_str_single() -> None:
    result = str(AnalysisList([Analysis("style analysis")]))
    assert result == "AnalysisList(\n  (0): Analysis(content_len=14, metadata=None)\n)"


def test_analysis_list_str_multiple() -> None:
    assert (
        str(AnalysisList([Analysis("style analysis"), Analysis("semantic analysis")]))
        == "AnalysisList(\n"
        "  (0): Analysis(content_len=14, metadata=None)\n"
        "  (1): Analysis(content_len=17, metadata=None)\n"
        ")"
    )


def test_analysis_list_str_nested() -> None:
    assert (
        str(
            AnalysisList(
                [
                    Analysis("style analysis"),
                    Analysis("semantic analysis"),
                    AnalysisList(
                        [
                            Analysis("I am a cat"),
                            Analysis("I am a bear"),
                        ]
                    ),
                ]
            )
        )
        == "AnalysisList(\n"
        "  (0): Analysis(content_len=14, metadata=None)\n"
        "  (1): Analysis(content_len=17, metadata=None)\n"
        "  (2): AnalysisList(\n"
        "      (0): Analysis(content_len=10, metadata=None)\n"
        "      (1): Analysis(content_len=11, metadata=None)\n"
        "    )\n"
        ")"
    )


def test_analysis_list_equal_true() -> None:
    assert AnalysisList([Analysis("style analysis")]).equal(
        AnalysisList([Analysis("style analysis")])
    )


def test_analysis_list_equal_true_empty() -> None:
    assert AnalysisList({}).equal(AnalysisList({}))


def test_analysis_list_equal_false_different_keys() -> None:
    assert not AnalysisList({"style": Analysis("my analysis")}).equal(
        AnalysisList({"semantic": Analysis("my analysis")})
    )


def test_analysis_list_equal_false_different_values() -> None:
    assert not AnalysisList([Analysis("style analysis")]).equal(
        AnalysisList({"style": Analysis("other analysis")})
    )


def test_analysis_list_equal_false_different_number_of_analyses() -> None:
    assert not AnalysisList([Analysis("style analysis")]).equal(
        AnalysisList(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
            }
        )
    )


def test_analysis_list_equal_false_different_type() -> None:
    assert not AnalysisList({}).equal({})


def test_analysis_list_equal_nan_false_by_default() -> None:
    assert not AnalysisList([Analysis(float("nan"))]).equal(AnalysisList([Analysis(float("nan"))]))


def test_analysis_list_equal_nan_true() -> None:
    assert AnalysisList([Analysis(float("nan"))]).equal(
        AnalysisList([Analysis(float("nan"))]), equal_nan=True
    )


def test_analysis_list_from_dict() -> None:
    analyses = [Analysis("style analysis")]
    assert AnalysisList.from_dict({"analyses": analyses}).equal(AnalysisList(analyses))


def test_analysis_list_from_dict_empty() -> None:
    assert AnalysisList.from_dict({"analyses": {}}).equal(AnalysisList({}))


def test_analysis_list_to_dict_empty() -> None:
    assert AnalysisList({}).to_dict() == {"analyses": []}


def test_analysis_list_to_dict_single() -> None:
    assert AnalysisList([Analysis("style analysis")]).to_dict() == {
        "analyses": [Analysis("style analysis")]
    }


def test_analysis_list_to_dict_multiple() -> None:
    assert AnalysisList([Analysis("style analysis"), Analysis("semantic analysis")]).to_dict() == {
        "analyses": [Analysis("style analysis"), Analysis("semantic analysis")]
    }


def test_analysis_list_to_dict_roundtrip() -> None:
    analysis = AnalysisList([Analysis("style analysis"), Analysis("semantic analysis")])
    assert AnalysisList.from_dict(analysis.to_dict()).equal(analysis)


def test_analysis_list_to_text_empty() -> None:
    assert AnalysisList({}).to_text() == "[]"


def test_analysis_list_to_text_single() -> None:
    assert AnalysisList([Analysis("style analysis")]).to_text() == "['style analysis']"


def test_analysis_list_to_text_multiple() -> None:
    assert (
        AnalysisList([Analysis("style analysis"), Analysis("semantic analysis")]).to_text()
        == "['style analysis', 'semantic analysis']"
    )


def test_analysis_list_to_text_nested() -> None:
    assert (
        AnalysisList(
            [
                Analysis("style analysis"),
                Analysis("semantic analysis"),
                AnalysisList(
                    [
                        Analysis("I am a cat"),
                        Analysis("I am a bear"),
                    ]
                ),
            ]
        ).to_text()
        == "['style analysis', 'semantic analysis', \"['I am a cat', 'I am a bear']\"]"
    )


##############################################
#     Tests for IndentedListAnalysisList     #
##############################################


def test_indented_list_analysis_list_to_text_empty() -> None:
    assert IndentedListAnalysisList({}).to_text() == ""


def test_indented_list_analysis_list_to_text_single() -> None:
    assert IndentedListAnalysisList([Analysis("style analysis")]).to_text() == ("- style analysis")


def test_indented_list_analysis_list_to_text_multiple() -> None:
    assert (
        IndentedListAnalysisList(
            [Analysis("style analysis"), Analysis("semantic analysis")]
        ).to_text()
        == "- style analysis\n- semantic analysis"
    )


def test_indented_list_analysis_list_to_text_nested() -> None:
    assert (
        IndentedListAnalysisList(
            [
                Analysis("style analysis"),
                Analysis("semantic analysis"),
                IndentedListAnalysisList(
                    [
                        Analysis("I am a cat"),
                        Analysis("I am a bear"),
                    ]
                ),
            ]
        ).to_text()
        == "- style analysis\n"
        "- semantic analysis\n"
        "- - I am a cat\n"
        "  - I am a bear"
    )
