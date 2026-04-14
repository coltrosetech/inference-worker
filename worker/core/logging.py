from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog


class _NamedPrintLogger:
    """Wrapper around PrintLogger that provides a .name attribute."""

    def __init__(self, name: str, file: Any = None):
        self.name = name
        self._logger = structlog.PrintLogger(file=file)

    def msg(self, message: str) -> None:
        self._logger.msg(message)

    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def critical(self, message: str) -> None:
        self._logger.critical(message)


class _NamedPrintLoggerFactory:
    """Factory that creates NamedPrintLogger instances."""

    def __init__(self, file: Any = None):
        self.file = file

    def __call__(self, *args: Any) -> _NamedPrintLogger:
        # First positional argument is the logger name if provided
        name = args[0] if args else "root"
        return _NamedPrintLogger(name, file=self.file)


def configure_logging(
    level: str = "info",
    fmt: Literal["json", "text"] = "json",
) -> None:
    """Configure structlog for the process.

    Call once at startup. Safe to call again (idempotent within a process).
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=_NamedPrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
