"""Flying the drone from the ground to its hover position."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import rclpy

from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.control.takeoff import TakeoffTrajectory
from crazyflow_experiments.utils.ros import calibrate_async

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from crazyflow_experiments.deploy.crazyflie import Drone

logger = logging.getLogger(__name__)


def takeoff(
    drone: Drone,
    target_pos: NDArray[np.floating],
    target_yaw: float,
    calibrate: bool = True,
    calibration_delay: float = 0.5,
    ctrl_freq: float = 50.0,
    duration: float = 4.0,
    hover_time: float = 2.0,
):
    """Fly the drone up to its starting position, hover there and calibrate it if requested.

    Args:
        drone: Drone interface.
        target_pos: Position to hover at as (3,).
        target_yaw: Yaw to hover at in radians.
        calibrate: Calibrate the drone orientation once it hovers.
        calibration_delay: Hover settle time for calibration.
        ctrl_freq: Rate at which the setpoints are sent.
        duration: Duration of the takeoff.
        hover_time: Duration of the hover before the calibration starts.
    """
    obs = drone.obs()
    trajectory = TakeoffTrajectory(obs["pos"], obs["rpy"][2], target_pos, target_yaw, duration)
    trajectory_pos, trajectory_yaw = trajectory.setpoints(ctrl_freq, hover_time)
    zeros = np.zeros(3)
    calibration_ready, tick = None, 0

    while rclpy.ok():
        t = time.perf_counter()
        obs = drone.obs()
        if obs["is_outdated"]:
            logger.warning("Measurements outdated. Emergency stop.")
            return

        pos, yaw = trajectory_pos[tick], trajectory_yaw[tick]
        drone.send_state(pos, zeros, zeros, yaw)
        tick = min(tick + 1, len(trajectory_pos) - 1)

        if tick == len(trajectory_pos) - 1:
            if not calibrate:
                logger.info("Takeoff finished. Starting experiment.")
                return
            if calibration_ready is None:
                logger.info("Takeoff finished. Starting calibration...")
                calibration_ready = calibrate_async(drone.id, delay=calibration_delay)
        if calibration_ready is not None and calibration_ready.is_set():
            logger.info("Calibration complete. Starting experiment.")
            return

        drone.update_estimators()
        sleep_until_next_cycle(t, ctrl_freq)
