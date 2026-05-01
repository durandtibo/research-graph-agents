from __future__ import annotations

from coola.equality import objects_are_equal

from argos.meta_agent.result import Result, ResultDict

################################
#     Tests for ResultDict     #
################################


def test_result_dict_repr_empty() -> None:
    assert repr(ResultDict({})) == "ResultDict(count=0)"


def test_result_dict_repr_single() -> None:
    assert repr(ResultDict({"train": Result({"loss": 0.5})})) == "ResultDict(count=1)"


def test_result_dict_repr_multiple() -> None:
    result = ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
    assert repr(result) == "ResultDict(count=2)"


def test_result_dict_str_empty() -> None:
    assert str(ResultDict({})) == "ResultDict()"


def test_result_dict_str_single() -> None:
    result = ResultDict({"train": Result({"loss": 0.5})})
    assert str(result) == "ResultDict(\n  (train): Result(loss=0.5)\n)"


def test_result_dict_str_multiple() -> None:
    result = ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
    assert str(result) == "ResultDict(\n  (train): Result(loss=0.5)\n  (val): Result(loss=0.3)\n)"


def test_result_dict_equal_true() -> None:
    assert ResultDict({"train": Result({"loss": 0.5})}).equal(
        ResultDict({"train": Result({"loss": 0.5})})
    )


def test_result_dict_equal_empty() -> None:
    assert ResultDict({}).equal(ResultDict({}))


def test_result_dict_equal_false_different_keys() -> None:
    assert not ResultDict({"train": Result({"loss": 0.5})}).equal(
        ResultDict({"val": Result({"loss": 0.5})})
    )


def test_result_dict_equal_false_different_values() -> None:
    assert not ResultDict({"train": Result({"loss": 0.5})}).equal(
        ResultDict({"train": Result({"loss": 0.9})})
    )


def test_result_dict_equal_false_different_type() -> None:
    assert not ResultDict({}).equal({})


def test_result_dict_equal_false_different_type_child() -> None:
    class Child(Result): ...

    assert not ResultDict({}).equal(Child({}))


def test_result_dict_equal_nan_false_by_default() -> None:
    assert not ResultDict({"train": Result({"loss": float("nan")})}).equal(
        ResultDict({"train": Result({"loss": float("nan")})})
    )


def test_result_dict_equal_nan_true() -> None:
    assert ResultDict({"train": Result({"loss": float("nan")})}).equal(
        ResultDict({"train": Result({"loss": float("nan")})}), equal_nan=True
    )


def test_result_dict_to_dict_empty() -> None:
    assert ResultDict({}).to_dict() == {}


def test_result_dict_to_dict() -> None:
    assert ResultDict({"train": Result({"loss": 0.5})}).to_dict() == {"train": {"loss": 0.5}}


def test_result_dict_to_dict_multiple() -> None:
    result = ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
    assert result.to_dict() == {"train": {"loss": 0.5}, "val": {"loss": 0.3}}


def test_result_dict_to_flat_dict_empty() -> None:
    assert ResultDict({}).to_flat_dict() == {}


def test_result_dict_to_flat_dict() -> None:
    result = ResultDict({"train": Result({"loss": 0.5})})
    assert result.to_flat_dict() == {"train.loss": 0.5}


def test_result_dict_to_flat_dict_multiple() -> None:
    result = ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
    assert result.to_flat_dict() == {"train.loss": 0.5, "val.loss": 0.3}


def test_result_dict_to_flat_dict_custom_separator() -> None:
    result = ResultDict({"train": Result({"loss": 0.5})})
    assert result.to_flat_dict(separator="/") == {"train/loss": 0.5}


def test_result_dict_to_raw_dict_empty() -> None:
    assert ResultDict({}).to_raw_dict() == {}


def test_result_dict_to_raw_dict() -> None:
    assert objects_are_equal(
        ResultDict({"train": Result({"loss": 0.5})}).to_raw_dict(), {"train": Result({"loss": 0.5})}
    )


def test_result_dict_to_raw_dict_returns_original_objects() -> None:
    child = Result({"loss": 0.5})
    assert ResultDict({"train": child}).to_raw_dict()["train"] is child


def test_result_dict_to_markdown_empty() -> None:
    assert ResultDict({}).to_markdown() == "_No metrics available._"


def test_result_dict_to_markdown_single() -> None:
    result = ResultDict({"train": Result({"loss": 0.5})})
    assert result.to_markdown() == "- **train**:\n  - **loss**: 0.5"


def test_result_dict_to_markdown_multiple() -> None:
    result = ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
    assert result.to_markdown() == (
        "- **train**:\n  - **loss**: 0.5\n- **val**:\n  - **loss**: 0.3"
    )
