"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure the root ``inky`` logger and return it.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger("inky")

    logger = logging.getLogger("inky")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "inky") -> logging.Logger:
    """Get a child logger under the ``inky`` namespace."""
    return logging.getLogger(f"inky.{name}")
