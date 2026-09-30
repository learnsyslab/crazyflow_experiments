"""Helpers shared by the real drone deployment scripts."""

import logging
import subprocess
import threading
import time

logger = logging.getLogger(__name__)


def _call_service(drone_id: int, service: str, timeout: float = 30.0):
    """Call a service of the drone estimator node and block until it returns.

    Args:
        drone_id: ID of the drone the estimator node is running for.
        service: Service name under the drone's namespace.
        timeout: Seconds to wait for the call. `ros2 service call` waits for a missing service
            forever, so without a timeout a wrong drone ID stalls the script silently.
    """
    name = f"/drones/cf{drone_id}/{service}"
    logger.info(f"Calling {name}...")
    cmd = ["ros2", "service", "call", name, "std_srvs/srv/Trigger"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
    response = result.stdout.strip().replace("\n", " | ")
    logger.info(f"Service response: {response}")


def clear_calibration(drone_id: int):
    """Drop the drone estimator orientation calibration."""
    _call_service(drone_id, "remove_calibration")


def calibrate_async(drone_id: int, delay: float = 0.0) -> threading.Event:
    """Calibrate the drone orientation in the background while the control loop keeps running.

    The service call blocks for several seconds, which would stall the loop and drop the drone.

    Args:
        drone_id: ID of the drone the estimator node is running for.
        delay: Seconds to wait before calling, to let the drone settle into a clean hover.

    Returns:
        An event that is set once the calibration has returned.
    """
    done = threading.Event()

    def run():
        time.sleep(delay)
        _call_service(drone_id, "calibration")
        done.set()

    threading.Thread(target=run, daemon=True).start()
    return done
