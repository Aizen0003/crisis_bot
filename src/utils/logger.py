"""
Structured logging for the Crisis Intelligence system.
Logs to stderr to avoid interfering with stdout-based tools.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Use stderr to avoid interfering with command runners / pipes
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger
