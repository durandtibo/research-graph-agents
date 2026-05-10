from __future__ import annotations

from argos.meta_agent.analyses import (
    Analysis,
    JsonAnalysisDict,
    YamlAnalysisDict,
)

######################################
#     Tests for JsonAnalysisDict     #
######################################


def test_json_analysis_dict_repr_empty() -> None:
    assert repr(JsonAnalysisDict({})) == "JsonAnalysisDict()"


def test_json_analysis_dict_repr_single() -> None:
    assert (
        repr(JsonAnalysisDict({"style": Analysis("style analysis")}))
        == "JsonAnalysisDict(num_analyses=1, indent=None, sort_keys=False)"
    )


def test_json_analysis_dict_repr_multiple() -> None:
    assert (
        repr(
            JsonAnalysisDict(
                {
                    "style": Analysis("style analysis"),
                    "semantic": Analysis("semantic analysis"),
                }
            )
        )
        == "JsonAnalysisDict(num_analyses=2, indent=None, sort_keys=False)"
    )


def test_json_analysis_dict_repr_nested() -> None:
    assert (
        repr(
            JsonAnalysisDict(
                {
                    "style": Analysis("style analysis"),
                    "semantic": Analysis("semantic analysis"),
                    "other": JsonAnalysisDict(
                        {
                            "cat": Analysis("I am a cat"),
                            "bear": Analysis("I am a bear"),
                        }
                    ),
                }
            )
        )
        == "JsonAnalysisDict(num_analyses=3, indent=None, sort_keys=False)"
    )


def test_json_analysis_dict_str_empty() -> None:
    assert str(JsonAnalysisDict({})) == "JsonAnalysisDict()"


def test_json_analysis_dict_str_single() -> None:
    assert (
        str(JsonAnalysisDict({"style": Analysis("style analysis")})) == "JsonAnalysisDict(\n"
        "  (indent): None\n"
        "  (sort_keys): False\n"
        "  (analyses): \n"
        "      (style): Analysis(content_len=14, metadata=None)\n"
        ")"
    )


def test_json_analysis_dict_str_multiple() -> None:
    assert (
        str(
            JsonAnalysisDict(
                {
                    "style": Analysis("style analysis"),
                    "semantic": Analysis("semantic analysis"),
                }
            )
        )
        == "JsonAnalysisDict(\n"
        "  (indent): None\n"
        "  (sort_keys): False\n"
        "  (analyses): \n"
        "      (style): Analysis(content_len=14, metadata=None)\n"
        "      (semantic): Analysis(content_len=17, metadata=None)\n"
        ")"
    )


def test_json_analysis_dict_str_nested() -> None:
    assert (
        str(
            JsonAnalysisDict(
                {
                    "style": Analysis("style analysis"),
                    "semantic": Analysis("semantic analysis"),
                    "other": JsonAnalysisDict(
                        {
                            "cat": Analysis("I am a cat"),
                            "bear": Analysis("I am a bear"),
                        }
                    ),
                }
            )
        )
        == "JsonAnalysisDict(\n"
        "  (indent): None\n"
        "  (sort_keys): False\n"
        "  (analyses): \n"
        "      (style): Analysis(content_len=14, metadata=None)\n"
        "      (semantic): Analysis(content_len=17, metadata=None)\n"
        "      (other): JsonAnalysisDict(\n"
        "          (indent): None\n"
        "          (sort_keys): False\n"
        "          (analyses): \n"
        "              (cat): Analysis(content_len=10, metadata=None)\n"
        "              (bear): Analysis(content_len=11, metadata=None)\n"
        "        )"
        "\n)"
    )


def test_json_analysis_dict_equal_true() -> None:
    assert JsonAnalysisDict({"style": Analysis("style analysis")}).equal(
        JsonAnalysisDict({"style": Analysis("style analysis")})
    )


def test_json_analysis_dict_equal_true_empty() -> None:
    assert JsonAnalysisDict({}).equal(JsonAnalysisDict({}))


def test_json_analysis_dict_equal_false_different_keys() -> None:
    assert not JsonAnalysisDict({"style": Analysis("my analysis")}).equal(
        JsonAnalysisDict({"semantic": Analysis("my analysis")})
    )


def test_json_analysis_dict_equal_false_different_values() -> None:
    assert not JsonAnalysisDict({"style": Analysis("style analysis")}).equal(
        JsonAnalysisDict({"style": Analysis("other analysis")})
    )


def test_json_analysis_dict_equal_false_different_number_of_analyses() -> None:
    assert not JsonAnalysisDict({"style": Analysis("style analysis")}).equal(
        JsonAnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
            }
        )
    )


def test_json_analysis_dict_equal_false_different_indent() -> None:
    assert not JsonAnalysisDict({"style": Analysis("my analysis")}).equal(
        JsonAnalysisDict({"style": Analysis("my analysis")}, indent=2)
    )


def test_json_analysis_dict_equal_false_different_sort_keys() -> None:
    assert not JsonAnalysisDict({"style": Analysis("my analysis")}).equal(
        JsonAnalysisDict({"style": Analysis("my analysis")}, sort_keys=True)
    )


def test_json_analysis_dict_equal_false_different_type() -> None:
    assert not JsonAnalysisDict({}).equal({})


def test_json_analysis_dict_equal_nan_false_by_default() -> None:
    assert not JsonAnalysisDict({"style": Analysis(float("nan"))}).equal(
        JsonAnalysisDict({"style": Analysis(float("nan"))})
    )


def test_json_analysis_dict_equal_nan_true() -> None:
    assert JsonAnalysisDict({"style": Analysis(float("nan"))}).equal(
        JsonAnalysisDict({"style": Analysis(float("nan"))}), equal_nan=True
    )


def test_json_analysis_dict_from_dict() -> None:
    analyses = {"style": Analysis("style analysis")}
    assert JsonAnalysisDict.from_dict({"analyses": analyses}).equal(JsonAnalysisDict(analyses))


def test_json_analysis_dict_from_dict_empty() -> None:
    assert JsonAnalysisDict.from_dict({"analyses": {}}).equal(JsonAnalysisDict({}))


def test_json_analysis_dict_to_dict_empty() -> None:
    assert JsonAnalysisDict({}).to_dict() == {"analyses": {}, "indent": None, "sort_keys": False}


def test_json_analysis_dict_to_dict_single() -> None:
    assert JsonAnalysisDict({"style": Analysis("style analysis")}).to_dict() == {
        "analyses": {"style": Analysis("style analysis")},
        "indent": None,
        "sort_keys": False,
    }


def test_json_analysis_dict_to_dict_multiple() -> None:
    assert JsonAnalysisDict(
        {
            "style": Analysis("style analysis"),
            "semantic": Analysis("semantic analysis"),
        }
    ).to_dict() == {
        "analyses": {
            "style": Analysis("style analysis"),
            "semantic": Analysis("semantic analysis"),
        },
        "indent": None,
        "sort_keys": False,
    }


def test_json_analysis_dict_to_dict_indent() -> None:
    assert JsonAnalysisDict({"style": Analysis("style analysis")}, indent=2).to_dict() == {
        "analyses": {"style": Analysis("style analysis")},
        "indent": 2,
        "sort_keys": False,
    }


def test_json_analysis_dict_to_dict_sort_keys() -> None:
    assert JsonAnalysisDict({"style": Analysis("style analysis")}, sort_keys=True).to_dict() == {
        "analyses": {"style": Analysis("style analysis")},
        "indent": None,
        "sort_keys": True,
    }


def test_json_analysis_dict_to_dict_roundtrip() -> None:
    analysis = JsonAnalysisDict(
        {
            "style": Analysis("style analysis"),
            "semantic": Analysis("semantic analysis"),
        },
        indent=2,
        sort_keys=True,
    )
    assert JsonAnalysisDict.from_dict(analysis.to_dict()).equal(analysis)


def test_json_analysis_dict_to_text_empty() -> None:
    assert JsonAnalysisDict({}).to_text() == "{}"


def test_json_analysis_dict_to_text_single() -> None:
    assert JsonAnalysisDict({"style": Analysis("style analysis")}).to_text() == (
        '{"style": "style analysis"}'
    )


def test_json_analysis_dict_to_text_multiple() -> None:
    assert (
        JsonAnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
            }
        ).to_text()
        == '{"style": "style analysis", "semantic": "semantic analysis"}'
    )


def test_json_analysis_dict_to_text_nested() -> None:
    assert (
        JsonAnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
                "other": JsonAnalysisDict(
                    {
                        "cat": Analysis("I am a cat"),
                        "bear": Analysis("I am a bear"),
                    }
                ),
            }
        ).to_text()
        == r'{"style": "style analysis", "semantic": "semantic analysis", '
        r'"other": "{\"cat\": \"I am a cat\", \"bear\": \"I am a bear\"}"}'
    )


def test_json_analysis_dict_to_text_indent_2() -> None:
    assert (
        JsonAnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
                "other": JsonAnalysisDict(
                    {
                        "cat": Analysis("I am a cat"),
                        "bear": Analysis("I am a bear"),
                    }
                ),
            },
            indent=2,
        ).to_text()
        == r"""{
  "style": "style analysis",
  "semantic": "semantic analysis",
  "other": "{\"cat\": \"I am a cat\", \"bear\": \"I am a bear\"}"
}"""
    )


######################################
#     Tests for YamlAnalysisDict     #
######################################


def test_yaml_analysis_dict_repr_empty() -> None:
    assert repr(YamlAnalysisDict({})) == "YamlAnalysisDict()"


def test_yaml_analysis_dict_repr_single() -> None:
    assert (
        repr(YamlAnalysisDict({"style": Analysis("style analysis")}))
        == "YamlAnalysisDict(num_analyses=1, indent=None, sort_keys=False)"
    )


def test_yaml_analysis_dict_repr_multiple() -> None:
    assert (
        repr(
            YamlAnalysisDict(
                {
                    "style": Analysis("style analysis"),
                    "semantic": Analysis("semantic analysis"),
                }
            )
        )
        == "YamlAnalysisDict(num_analyses=2, indent=None, sort_keys=False)"
    )


def test_yaml_analysis_dict_repr_nested() -> None:
    assert (
        repr(
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
            )
        )
        == "YamlAnalysisDict(num_analyses=3, indent=None, sort_keys=False)"
    )


def test_yaml_analysis_dict_str_empty() -> None:
    assert str(YamlAnalysisDict({})) == "YamlAnalysisDict()"


def test_yaml_analysis_dict_str_single() -> None:
    assert (
        str(YamlAnalysisDict({"style": Analysis("style analysis")})) == "YamlAnalysisDict(\n"
        "  (indent): None\n"
        "  (sort_keys): False\n"
        "  (analyses): \n"
        "      (style): Analysis(content_len=14, metadata=None)\n"
        ")"
    )


def test_yaml_analysis_dict_str_multiple() -> None:
    assert (
        str(
            YamlAnalysisDict(
                {
                    "style": Analysis("style analysis"),
                    "semantic": Analysis("semantic analysis"),
                }
            )
        )
        == "YamlAnalysisDict(\n"
        "  (indent): None\n"
        "  (sort_keys): False\n"
        "  (analyses): \n"
        "      (style): Analysis(content_len=14, metadata=None)\n"
        "      (semantic): Analysis(content_len=17, metadata=None)\n"
        ")"
    )


def test_yaml_analysis_dict_str_nested() -> None:
    assert (
        str(
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
            )
        )
        == "YamlAnalysisDict(\n"
        "  (indent): None\n"
        "  (sort_keys): False\n"
        "  (analyses): \n"
        "      (style): Analysis(content_len=14, metadata=None)\n"
        "      (semantic): Analysis(content_len=17, metadata=None)\n"
        "      (other): YamlAnalysisDict(\n"
        "          (indent): None\n"
        "          (sort_keys): False\n"
        "          (analyses): \n"
        "              (cat): Analysis(content_len=10, metadata=None)\n"
        "              (bear): Analysis(content_len=11, metadata=None)\n"
        "        )"
        "\n)"
    )


def test_yaml_analysis_dict_equal_true() -> None:
    assert YamlAnalysisDict({"style": Analysis("style analysis")}).equal(
        YamlAnalysisDict({"style": Analysis("style analysis")})
    )


def test_yaml_analysis_dict_equal_true_empty() -> None:
    assert YamlAnalysisDict({}).equal(YamlAnalysisDict({}))


def test_yaml_analysis_dict_equal_false_different_keys() -> None:
    assert not YamlAnalysisDict({"style": Analysis("my analysis")}).equal(
        YamlAnalysisDict({"semantic": Analysis("my analysis")})
    )


def test_yaml_analysis_dict_equal_false_different_values() -> None:
    assert not YamlAnalysisDict({"style": Analysis("style analysis")}).equal(
        YamlAnalysisDict({"style": Analysis("other analysis")})
    )


def test_yaml_analysis_dict_equal_false_different_number_of_analyses() -> None:
    assert not YamlAnalysisDict({"style": Analysis("style analysis")}).equal(
        YamlAnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
            }
        )
    )


def test_yaml_analysis_dict_equal_false_different_indent() -> None:
    assert not YamlAnalysisDict({"style": Analysis("my analysis")}).equal(
        YamlAnalysisDict({"style": Analysis("my analysis")}, indent=2)
    )


def test_yaml_analysis_dict_equal_false_different_sort_keys() -> None:
    assert not YamlAnalysisDict({"style": Analysis("my analysis")}).equal(
        YamlAnalysisDict({"style": Analysis("my analysis")}, sort_keys=True)
    )


def test_yaml_analysis_dict_equal_false_different_type() -> None:
    assert not YamlAnalysisDict({}).equal({})


def test_yaml_analysis_dict_equal_nan_false_by_default() -> None:
    assert not YamlAnalysisDict({"style": Analysis(float("nan"))}).equal(
        YamlAnalysisDict({"style": Analysis(float("nan"))})
    )


def test_yaml_analysis_dict_equal_nan_true() -> None:
    assert YamlAnalysisDict({"style": Analysis(float("nan"))}).equal(
        YamlAnalysisDict({"style": Analysis(float("nan"))}), equal_nan=True
    )


def test_yaml_analysis_dict_from_dict() -> None:
    analyses = {"style": Analysis("style analysis")}
    assert YamlAnalysisDict.from_dict({"analyses": analyses}).equal(YamlAnalysisDict(analyses))


def test_yaml_analysis_dict_from_dict_empty() -> None:
    assert YamlAnalysisDict.from_dict({"analyses": {}}).equal(YamlAnalysisDict({}))


def test_yaml_analysis_dict_to_dict_empty() -> None:
    assert YamlAnalysisDict({}).to_dict() == {"analyses": {}, "indent": None, "sort_keys": False}


def test_yaml_analysis_dict_to_dict_single() -> None:
    assert YamlAnalysisDict({"style": Analysis("style analysis")}).to_dict() == {
        "analyses": {"style": Analysis("style analysis")},
        "indent": None,
        "sort_keys": False,
    }


def test_yaml_analysis_dict_to_dict_multiple() -> None:
    assert YamlAnalysisDict(
        {
            "style": Analysis("style analysis"),
            "semantic": Analysis("semantic analysis"),
        }
    ).to_dict() == {
        "analyses": {
            "style": Analysis("style analysis"),
            "semantic": Analysis("semantic analysis"),
        },
        "indent": None,
        "sort_keys": False,
    }


def test_yaml_analysis_dict_to_dict_indent() -> None:
    assert YamlAnalysisDict({"style": Analysis("style analysis")}, indent=2).to_dict() == {
        "analyses": {"style": Analysis("style analysis")},
        "indent": 2,
        "sort_keys": False,
    }


def test_yaml_analysis_dict_to_dict_sort_keys() -> None:
    assert YamlAnalysisDict({"style": Analysis("style analysis")}, sort_keys=True).to_dict() == {
        "analyses": {"style": Analysis("style analysis")},
        "indent": None,
        "sort_keys": True,
    }


def test_yaml_analysis_dict_to_dict_roundtrip() -> None:
    analysis = YamlAnalysisDict(
        {
            "style": Analysis("style analysis"),
            "semantic": Analysis("semantic analysis"),
        },
        indent=2,
        sort_keys=True,
    )
    assert YamlAnalysisDict.from_dict(analysis.to_dict()).equal(analysis)


def test_yaml_analysis_dict_to_text_empty() -> None:
    assert YamlAnalysisDict({}).to_text() == "{}\n"


def test_yaml_analysis_dict_to_text_single() -> None:
    assert YamlAnalysisDict({"style": Analysis("style analysis")}).to_text() == (
        "style: style analysis\n"
    )


def test_yaml_analysis_dict_to_text_multiple() -> None:
    assert (
        YamlAnalysisDict(
            {
                "style": Analysis("style analysis"),
                "semantic": Analysis("semantic analysis"),
            }
        ).to_text()
        == "style: style analysis\nsemantic: semantic analysis\n"
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
        "other:"
        "  cat: I am a cat\n"
        "  bear: I am a bear\n"
    )


def test_yaml_analysis_dict_to_text_nested_indent_4() -> None:
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
            },
            indent=4,
        ).to_text()
        == "style: style analysis\n"
        "semantic: semantic analysis\n"
        "other:\n"
        "    cat: I am a cat\n"
        "    bear: I am a bear\n"
    )
