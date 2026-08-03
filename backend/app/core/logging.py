"""JSON structured logging with request-context enrichment."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import Settings


class JsonFormatter(logging.Formatter):
    """Serialize standard log records into machine-readable JSON lines."""

    _reserved = frozenset(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._reserved and key not in {"message", "asctime"}:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings) -> None:
    """Configure root logging once, writing structured output to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = (
        JsonFormatter()
        if settings.log_format == "json"
        else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a named logger following the centralized configuration."""
    return logging.getLogger(name)
