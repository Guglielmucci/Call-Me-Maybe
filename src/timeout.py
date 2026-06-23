"""Timeout context manager using SIGALRM (Unix only)."""
from __future__ import annotations

import signal
import math
import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def time_limit(seconds: float) -> Generator[None, None, None]:
    """Context manager that limits execution time using SIGALRM (Unix only).

    On Windows or if SIGALRM is not available, the manager is a no‑op
    (no timeout).  Raises ``TimeoutError`` if the block exceeds the
    given time on Unix.

    Args:
        seconds: Maximum allowed wall‑clock time.
    """
    if not hasattr(signal, "SIGALRM"):
        logger.warning("SIGALRM not available (Windows?) Time limit disabled")
        yield
        return
    # Round the timeout up to at least 1 second to avoid alarm(0)
    alarm_seconds = max(1, math.ceil(seconds))

    def handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"Time limit exceeded ({seconds}s)")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(alarm_seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
