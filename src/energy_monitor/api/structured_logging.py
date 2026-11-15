import json
import logging
from datetime import UTC, datetime
from typing import Any, Final

LOGGER_NAME: Final = "energy_monitor"
STANDARD_ATTRIBUTES: Final = frozenset(vars(logging.makeLogRecord({})))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(
            {key: value for key, value in vars(record).items() if key not in STANDARD_ATTRIBUTES}
        )
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(handler.formatter, JsonFormatter) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
