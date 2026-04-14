import io
import json

import structlog

from worker.core.logging import configure_logging, get_logger


def test_configure_logging_emits_json(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    configure_logging(level="info", fmt="json")
    log = get_logger("test")
    log.info("hello", foo="bar", count=3)
    line = buf.getvalue().strip()
    payload = json.loads(line)
    assert payload["event"] == "hello"
    assert payload["foo"] == "bar"
    assert payload["count"] == 3
    assert payload["level"] == "info"
    assert payload["logger"] == "test"
    assert "timestamp" in payload


def test_configure_logging_text_format(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    configure_logging(level="info", fmt="text")
    log = get_logger("test")
    log.info("hello", foo="bar")
    line = buf.getvalue().strip()
    assert "hello" in line
    assert "foo=bar" in line
