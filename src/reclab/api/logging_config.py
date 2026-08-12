"""Structured (JSON-lines) logging for the API.

Plain stdlib `logging`, not a third-party structured-logging library — this
is a self-hosted, single-process tool; anything heavier (structlog, a
logging-service SDK) would be more infrastructure than the problem calls
for. JSON lines are still pipeable into `jq` or any real log aggregator
without extra glue.

Every event goes through `log_event(name, **fields)` rather than ad hoc
`logger.info(f"...")` calls, so every line has the same shape
(`timestamp`, `level`, `message`, plus whatever fields the call site
passed) instead of a mix of free-text and structured lines.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger("reclab")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", None) or {})
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Idempotent: safe to call from multiple import sites (main.py, tests)
    without duplicating handlers."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    # Don't also hand records up to the root logger — this is the only
    # handler we want driving output, so no duplicate lines from a
    # differently-configured root (e.g. uvicorn's own logging setup).
    logger.propagate = False


def log_event(event: str, **fields: Any) -> None:
    logger.info(event, extra={"extra_fields": fields})
