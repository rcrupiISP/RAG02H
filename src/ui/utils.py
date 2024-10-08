import logging
from typing import Callable, List, Type


class StreamlitLogHandler(logging.Handler):
    def __init__(self, widget_update_func: Callable[[str], None]) -> None:
        """Initializes the Streamlit log handler.

        Args:
            widget_update_func (Callable[[str], None]): Function to update the widget with log messages.
        """
        super().__init__()
        self.widget_update_func = widget_update_func
        self.log_buffer: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Processes a log record and updates the widget.

        Args:
            record (logging.LogRecord): The log record containing log information.
        """
        msg = self.format(record)
        self.log_buffer.append(msg)

        # Limit the log buffer to the last 15 messages
        if len(self.log_buffer) > 15:
            self.log_buffer.pop(0)

        self.widget_update_func("\n".join(self.log_buffer))


def create_log_handler(
    handler_class: Type[logging.Handler], formatter: logging.Formatter = None, *args
) -> logging.Handler:
    """Creates and returns a logging handler with an optional formatter.

    Args:
        handler_class (Type[logging.Handler]): The class of the logging handler to create.
        formatter (logging.Formatter, optional): An optional formatter for the handler.
        *args: Additional arguments to pass to the handler's constructor.

    Returns:
        logging.Handler: The configured logging handler.
    """
    handler = handler_class(*args)
    if formatter is not None:
        handler.setFormatter(formatter)
    return handler


def setup_logger(
    logger: logging.Logger, handlers: List[logging.Handler], level: str, propagate: bool
) -> None:
    """Sets up the logger with the specified handlers and configuration.

    Args:
        logger (logging.Logger): The logger to configure.
        handlers (List[logging.Handler]): The list of handlers to add to the logger.
        level (str): The logging level to set (e.g., 'DEBUG', 'INFO').
        propagate (bool): Whether to propagate messages to higher-level loggers.

    Returns:
        None
    """
    logger.handlers.clear()
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = propagate
