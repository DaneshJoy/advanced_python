import logging
import sys
from pathlib import Path

PACKAGE_NAME = "eeg_generator"

def setup_logging(
    level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: str | Path | None = None,
    fmt: str = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
):
    """
    Setup package-wide logger.

    Parameters
    ----------
    level : int
        Logging level (default: INFO)
    log_to_console : bool
        Whether to log to terminal (default: True)
    log_to_file : str or Path, optional
        File path to log to (default: None)
    fmt : str
        Log message format
    datefmt : str
        Datetime format
    """
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_path = Path(log_to_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None):
    """
    Return a logger in this package namespace. If name is None, returns the root package logger.
    """
    if name:
        return logging.getLogger(f"{PACKAGE_NAME}.{name}")
    return logging.getLogger(PACKAGE_NAME)