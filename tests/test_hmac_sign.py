import time

import pytest

from worker.core.hmac_sign import sign, verify


def test_sign_deterministic_with_fixed_timestamp():
    body = b'{"job_id":"x"}'
    header = sign(body, secret="topsecret1234567", now=1700000000)
    assert header.startswith("t=1700000000,")
    assert ",v1=" in header


def test_verify_roundtrip():
    body = b'{"job_id":"x"}'
    header = sign(body, secret="topsecret1234567", now=1700000000)
    assert verify(body, header, secret="topsecret1234567", now=1700000000) is True


def test_verify_rejects_tampered_body():
    body = b'{"job_id":"x"}'
    header = sign(body, secret="topsecret1234567", now=1700000000)
    assert verify(b'{"job_id":"y"}', header, secret="topsecret1234567", now=1700000000) is False


def test_verify_rejects_bad_secret():
    body = b'{"job_id":"x"}'
    header = sign(body, secret="topsecret1234567", now=1700000000)
    assert verify(body, header, secret="wrongsecret12345", now=1700000000) is False


def test_verify_rejects_expired_timestamp():
    body = b'{"job_id":"x"}'
    header = sign(body, secret="topsecret1234567", now=1700000000)
    assert verify(body, header, secret="topsecret1234567", now=1700000400, skew=300) is False


def test_verify_rejects_malformed_header():
    body = b'{"job_id":"x"}'
    assert verify(body, "garbage", secret="topsecret1234567", now=1700000000) is False
    assert verify(body, "t=abc,v1=deadbeef", secret="topsecret1234567", now=1700000000) is False
    assert verify(body, "t=1700000000", secret="topsecret1234567", now=1700000000) is False


def test_sign_default_now_is_current_time():
    body = b"payload"
    before = int(time.time())
    header = sign(body, secret="topsecret1234567")
    after = int(time.time())
    ts = int(header.split(",")[0].split("=")[1])
    assert before <= ts <= after
