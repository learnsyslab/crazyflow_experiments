"""Control loop utilities."""

import logging
import time

logger = logging.getLogger(__name__)


def sleep_until_next_cycle(cycle_start: float, ctrl_freq: float):
    """Sleep to keep the control cycle frequency.

    Args:
        cycle_start: Cycle start time. Must be from time.perf_counter().
        ctrl_freq: Control loop frequency.
    """
    remaining = 1 / ctrl_freq - (time.perf_counter() - cycle_start)
    if remaining < 0:
        logger.warning(f"Controller execution time exceeded loop frequency by {-remaining:.3f}s.")
        return
    time.sleep(remaining)
