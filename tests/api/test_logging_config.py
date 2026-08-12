from __future__ import annotations

import json
import logging

from reclab.api.logging_config import _JsonFormatter, configure_logging, log_event, logger


def test_log_event_emits_a_json_line(caplog):
    caplog.set_level(logging.INFO, logger="reclab")

    log_event("something_happened", job_id="abc123", duration_ms=12.5)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "something_happened"
    assert record.extra_fields == {"job_id": "abc123", "duration_ms": 12.5}


def test_json_formatter_produces_valid_json_with_expected_keys():
    record = logging.LogRecord(
        name="reclab",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="my_event",
        args=(),
        exc_info=None,
    )
    record.extra_fields = {"status_code": 200, "path": "/health"}

    formatted = _JsonFormatter().format(record)
    parsed = json.loads(formatted)

    assert parsed["message"] == "my_event"
    assert parsed["level"] == "INFO"
    assert parsed["status_code"] == 200
    assert parsed["path"] == "/health"
    assert "timestamp" in parsed


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()
    configure_logging()

    assert len(logger.handlers) == 1
    assert logger.propagate is False
