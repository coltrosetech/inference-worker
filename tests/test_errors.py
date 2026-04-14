import pytest

from worker.core.errors import AppError, ErrorCode


def test_app_error_carries_code_and_retryable():
    err = AppError(ErrorCode.OOM_CUDA, "out of memory", retryable=True)
    assert err.code is ErrorCode.OOM_CUDA
    assert err.retryable is True
    assert "out of memory" in str(err)


def test_app_error_default_retryable_false():
    err = AppError(ErrorCode.INPUT_FETCH_FAILED, "404")
    assert err.retryable is False


def test_app_error_to_dict_shape():
    err = AppError(ErrorCode.INFERENCE_FAILED, "node xyz crashed", retryable=True)
    d = err.to_dict()
    assert d == {"code": "INFERENCE_FAILED", "message": "node xyz crashed", "retryable": True}


def test_all_error_codes_are_stable_strings():
    for code in ErrorCode:
        assert code.value == code.name


def test_raise_and_catch():
    with pytest.raises(AppError) as exc:
        raise AppError(ErrorCode.JOB_TIMEOUT, "exceeded 300s", retryable=True)
    assert exc.value.code is ErrorCode.JOB_TIMEOUT
