"""Flight of an experiment on the real drone, shared by the deployment scripts.

The drone is a Crazyflie over the radio (`deploy/crazyflie.py`) or a PX4 vehicle through MAVROS
(`deploy/px4.py`), chosen by the drone model in the config. The scripts build their trajectory and
controller, then `connect`, `start` (take off, or take over from the pilot), `track` the controller
and `finish` (return home, or hold until the pilot takes over).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable

import rclpy

from crazyflow_experiments.control.config import drone_params
from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.deploy.crazyflie import Drone
from crazyflow_experiments.deploy.px4 import Drone as PX4Drone
from crazyflow_experiments.deploy.takeoff import takeoff
from crazyflow_experiments.utils.ros import clear_calibration

if TYPE_CHECKING:
    import numpy as np
    from crazyflow.control import Control
    from numpy.typing import NDArray

    from crazyflow_experiments.control.controller import Controller

logger = logging.getLogger(__name__)


def is_px4(config: dict) -> bool:
    """Whether the drone of the config is a PX4 vehicle rather than a Crazyflie."""
    return not config["drone"]["model"].startswith("cf")


def connect(
    config: dict,
    ctrl_interface: Control,
    log_variables: list[str] | None = None,
    log_callback: Callable | None = None,
) -> Drone | PX4Drone:
    """Connect to the drone the config describes.

    Args:
        config: Config whose `drone` section names the model, whose physical parameters come from
            `sysid/config_drones.toml`, with the radio settings of a Crazyflie under `drone.cf`,
            or in `drone` itself for experiments that only fly Crazyflies, and the MAVROS settings
            of a PX4 drone under `drone.px4`.
        ctrl_interface: Control interface the actions are sent through.
        log_variables: Firmware variables a Crazyflie streams back.
        log_callback: Receives the streamed variables.
    """
    model = config["drone"]["model"]
    if is_px4(config):
        px4, params = config["drone"]["px4"], drone_params(model)
        return PX4Drone(
            thrust_curve=params["thrust_curve"],
            thrust_min=params["thrust_min"],
            thrust_max=params["thrust_max"],
            ctrl_interface=ctrl_interface,
            namespace=px4["namespace"],
            name=model,
            localization=px4["localization"],
            stream_rate=px4["stream_rate"] or None,
        )
    # Experiments that only fly Crazyflies keep the radio settings in the drone section itself
    cf = config["drone"].get("cf", config["drone"])
    clear_calibration(cf["id"])  # Otherwise calibration offsets accumulate over multiple runs
    return Drone(
        drone_id=cf["id"],
        channel=cf["channel"],
        model=model,
        pos_landing=cf["pos_landing"],
        pos_update_freq=cf["pos_update_freq"],
        ctrl_interface=ctrl_interface,
        log_variables=log_variables,
        log_callback=log_callback,
    )


def start(
    drone: Drone | PX4Drone,
    pos_start: NDArray[np.floating],
    yaw_start: float,
    ctrl_freq: float,
    calibrate: bool = True,
):
    """Bring the drone to the start pose of the experiment and hover there.

    A Crazyflie takes off from the ground. A PX4 vehicle hovers under the pilot's control until the
    RC mode switch hands it over, then flies to the pose.

    Args:
        drone: Drone interface.
        pos_start: Position of the start pose as (3,).
        yaw_start: Yaw of the start pose in radians.
        ctrl_freq: Rate at which the setpoints are sent, the control frequency of the flight.
        calibrate: Calibrate the drone orientation once it hovers.
    """
    if isinstance(drone, PX4Drone):
        drone.wait_for_offboard()
        drone.fly_to(pos_start, yaw_start, ctrl_freq=ctrl_freq)
    else:
        takeoff(drone, pos_start, yaw_start, calibrate=calibrate, ctrl_freq=ctrl_freq)


def track(
    drone: Drone | PX4Drone, controller: Controller, ctrl_freq: float, start_time: float
) -> list[dict]:
    """Run the controller until it is done, the measurements drop out or the pilot takes over.

    A PX4 vehicle lands when the measurements drop out, see `deploy/px4.py`.

    Args:
        drone: Drone interface.
        controller: Controller flying the drone, it joins its trajectory now.
        ctrl_freq: Control loop frequency.
        start_time: Trajectory time the controller starts at, negative for a soft start that ramps
            in from a hover at the start pose.

    Returns:
        The observations of every control step with the action under "u", timed in trajectory
        time, zero where the trajectory starts.
    """
    px4 = isinstance(drone, PX4Drone)
    controller._t_start = time.perf_counter() - start_time
    data = []
    while rclpy.ok():
        t = time.perf_counter()
        obs = drone.obs()
        obs["t"] = t
        if obs["is_outdated"]:
            logger.warning("Measurements outdated. Emergency stop.")
            if px4:
                # Landing, not holding: the hold point comes from the estimate that just went stale
                drone.land()
                return data
            break
        if px4 and not drone.offboard:
            logger.info("The pilot took over. Stopping.")
            break
        action = controller(obs)
        drone.send_action(action)
        obs["u"] = action
        data.append(obs | {"t": t - controller._t_start})  # Recorded in trajectory time
        if controller.done(obs):
            logger.info("Controller finished.")
            break
        if not px4:
            drone.update_estimators()
        sleep_until_next_cycle(t, ctrl_freq)
    if px4:
        drone.hold()  # Right away, the last attitude setpoint is not repeated
    return data


def finish(drone: Drone | PX4Drone, keep_height: bool = False):
    """Hand the drone back: a Crazyflie returns home, a PX4 vehicle holds until the pilot takes over.

    Args:
        drone: The drone to hand back.
        keep_height: A Crazyflie returns at its current height, see `Drone.return_to_start`.
    """
    if isinstance(drone, PX4Drone):
        logger.info("done, holding position until the pilot takes over")
        drone.wait_for_takeover()
    else:
        drone.return_to_start(keep_height=keep_height)
