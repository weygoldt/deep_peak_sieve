import logging

from rich.logging import RichHandler

logger = logging.getLogger("thunderpulse")
DEFAULT_LOG_LEVEL = "DEBUG"


def setup_logging(logger, level=DEFAULT_LOG_LEVEL):
    if logger.hasHandlers():
        logger.handlers.clear()
    stream_handler = RichHandler(rich_tracebacks=True, show_path=False)
    stream_handler.setLevel(level)
    fmt_shell = "%(filename)s:%(lineno)d - %(message)s"
    # fmt_shell = "%(name)s: %(message)s"
    shell_formatter = logging.Formatter(fmt_shell)
    stream_handler.setFormatter(shell_formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(level)
