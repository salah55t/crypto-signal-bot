"""Logger utility - rich, colorful, persistent logging."""
import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from config.settings import settings, PROJECT_ROOT


def setup_logger(name: str = "crypto_bot") -> logging.Logger:
    """Configure and return a logger with both file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Rich console handler
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        log_time_format="[%H:%M:%S]",
    )
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler
    log_path = PROJECT_ROOT / settings.LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


# Global logger
log = setup_logger()
