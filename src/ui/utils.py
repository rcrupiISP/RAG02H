import logging
from typing import Type


class StreamlitLogHandler(logging.Handler):
    def __init__(self, widget_update_func):
        super().__init__()
        self.widget_update_func = widget_update_func
        self.log_buffer = []

    def emit(self, record):
        msg = self.format(record)
        self.log_buffer.append(msg)
        # Limita a 15 righe di log
        if len(self.log_buffer) > 15:
            self.log_buffer.pop(0)
        self.widget_update_func("\n".join(self.log_buffer))


def create_log_handler(
    handler_class: Type[logging.Handler], formatter: logging.Formatter = None, *args
) -> logging.Handler:
    out = handler_class(*args)
    if formatter is not None:
        out.setFormatter(formatter)
    return out


def setup_logger(
    logger: logging.Logger, handlers: list[logging.Handler], level: str, propagate: bool
) -> None:
    logger.handlers.clear()
    for h in handlers:
        logger.addHandler(h)
    logger.setLevel(level.upper())
    logger.propagate = propagate
