"""Fly a sim2real experiment on the real drone and record it.

Crazyflies are flown over the radio with their state from motion capture, see
`deploy/crazyflie.py`. Every other drone model is a PX4 vehicle flown through MAVROS on its
companion computer, see `deploy/px4.py`: the pilot takes off by hand, the script waits for the RC
mode switch to OFFBOARD, flies to the trajectory start, and holds position at the end until the
pilot takes over.

Usage:

    pixi run -e sim2real python crazyflow_experiments/sim2real/deploy.py

"""

import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import toml

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.interface import control_interface
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.control.state_controller import StateController
from crazyflow_experiments.deploy.experiment import connect, finish, start, track
from crazyflow_experiments.sim2real import RECORDINGS_DIR
from crazyflow_experiments.sim2real.utils import next_recording_path

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(calibrate: bool = True, save: bool = True):
    """Fly the configured trajectory with the configured controller."""
    config = toml.load(Path(__file__).parent / "config.toml")

    ctrl_type = config["controller"]["type"]
    ctrl_freq = config["controller"]["ctrl_freq"]
    ctrl_interface = control_interface(ctrl_type)
    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    pos_start, yaw_start = trajectory_config.start_pose(trajectory)

    model = config["drone"]["model"]

    initial_obs = {
        "pos": np.zeros(3),
        "quat": np.zeros(4),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "rpy": np.zeros(3),
        "t": time.perf_counter(),
    }
    match ctrl_type:
        case "mellinger":
            controller = MellingerController(
                initial_obs,
                trajectory,
                model,
                tilt_max=np.deg2rad(config["controller"]["tilt_max"]) or None,
            )
        case "state":
            controller = StateController(initial_obs, trajectory)
        case _:
            raise ValueError(f"Unknown controller type: {ctrl_type}")

    drone = connect(config, ctrl_interface)
    try:
        start(drone, pos_start, yaw_start, ctrl_freq, calibrate=calibrate)
        data = track(drone, controller, ctrl_freq, trajectory_config.start_time)
        finish(drone)
    finally:
        drone.close()

    if save:
        logger.info("Storing data...")
        with open(next_recording_path(RECORDINGS_DIR, config, deploy=True), "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    main()
