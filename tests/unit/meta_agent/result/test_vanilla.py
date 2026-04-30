from __future__ import annotations

from argos.meta_agent.result import Result

############################
#     Tests for Result     #
############################


def test_result_repr() -> None:
    assert repr(Result({"loss": 0.5})) == "Result(loss=0.5)"


def test_result_repr_empty() -> None:
    assert repr(Result({})) == "Result()"


def test_result_repr_multiple_metrics() -> None:
    assert repr(Result({"loss": 0.5, "accuracy": 0.9})) == "Result(loss=0.5, accuracy=0.9)"


def test_result_str() -> None:
    assert str(Result({"loss": 0.5})) == "Result(loss=0.5)"


def test_result_str_empty() -> None:
    assert str(Result({})) == "Result()"


def test_result_str_multiple_metrics() -> None:
    assert str(Result({"loss": 0.5, "accuracy": 0.9})) == "Result(loss=0.5, accuracy=0.9)"


def test_result_equal_true() -> None:
    assert Result({"loss": 0.5}).equal(Result({"loss": 0.5}))


def test_result_equal_false_different_values() -> None:
    assert not Result({"loss": 0.5}).equal(Result({"loss": 0.9}))


def test_result_equal_false_different_keys() -> None:
    assert not Result({"loss": 0.5}).equal(Result({"accuracy": 0.5}))


def test_result_equal_false_different_type() -> None:
    assert not Result({"loss": 0.5}).equal({"loss": 0.5})


def test_result_equal_false_different_type_child() -> None:
    class Child(Result): ...

    assert not Result({"loss": 0.5}).equal(Child({"loss": 0.5}))


def test_result_equal_empty() -> None:
    assert Result({}).equal(Result({}))


def test_result_equal_nan_false_by_default() -> None:
    assert not Result({"loss": float("nan")}).equal(Result({"loss": float("nan")}))


def test_result_equal_nan_true() -> None:
    assert Result({"loss": float("nan")}).equal(Result({"loss": float("nan")}), equal_nan=True)


def test_result_to_dict() -> None:
    assert Result({"loss": 0.5, "accuracy": 0.9}).to_dict() == {"loss": 0.5, "accuracy": 0.9}


def test_result_to_dict_empty() -> None:
    assert Result({}).to_dict() == {}


def test_result_to_dict_is_same_object_as_to_raw_dict() -> None:
    result = Result({"loss": 0.5, "accuracy": 0.9})
    assert result.to_dict() is not result.to_raw_dict()
    assert result.to_dict() == result.to_raw_dict()


def test_result_to_flat_dict() -> None:
    assert Result({"loss": 0.5, "accuracy": 0.9}).to_flat_dict() == {"loss": 0.5, "accuracy": 0.9}


def test_result_to_flat_dict_empty() -> None:
    assert Result({}).to_flat_dict() == {}


def test_result_to_flat_dict_custom_separator() -> None:
    assert Result({"loss": 0.5}).to_flat_dict(separator="/") == {"loss": 0.5}


def test_result_to_raw_dict() -> None:
    assert Result({"loss": 0.5, "accuracy": 0.9}).to_raw_dict() == {"loss": 0.5, "accuracy": 0.9}


def test_result_to_raw_dict_empty() -> None:
    assert Result({}).to_raw_dict() == {}


def test_result_to_raw_dict_returns_original_object() -> None:
    metrics = {"loss": 0.5}
    out = Result(metrics).to_raw_dict()
    assert out is not metrics
    assert out == metrics


def test_result_to_markdown_empty() -> None:
    assert Result({}).to_markdown() == "_No metrics available._"


def test_result_to_markdown_single_metric() -> None:
    assert Result({"loss": 0.5}).to_markdown() == "- **loss**: 0.5"


def test_result_to_markdown_multiple_metrics() -> None:
    assert (
        Result({"loss": 0.5, "accuracy": 0.9}).to_markdown()
        == "- **loss**: 0.5\n- **accuracy**: 0.9"
    )


def test_result_to_markdown_integer_value() -> None:
    assert Result({"epoch": 10}).to_markdown() == "- **epoch**: 10"


def test_result_to_markdown_string_value() -> None:
    assert Result({"model": "resnet50"}).to_markdown() == "- **model**: resnet50"
