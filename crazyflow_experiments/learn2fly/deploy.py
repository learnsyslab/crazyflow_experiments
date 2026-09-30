"""Fly the reach-position policy on the real Crazyflie and record the flight.

The firmware runs the exported TD3 actor, see `export.py`, as its controller 8 and takes the
reference state of the trajectory as its goal. The script therefore streams the trajectory through
the state interface and swaps to the onboard policy once the drone hovers at the start, see
`deploy/crazyflie.py` for the radio link. Start the tracker and the estimator first, see the main
README, then run

    pixi run -e learn2fly python crazyflow_experiments/learn2fly/deploy.py

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
from crazyflow_experiments.control.state_controller import StateController
from crazyflow_experiments.deploy.experiment import connect, track
from crazyflow_experiments.deploy.takeoff import takeoff
from crazyflow_experiments.learn2fly import RECORDINGS_DIR
from crazyflow_experiments.sim2real.utils import generate_filename_from_config
from crazyflow_experiments.utils.paths import new_recording_path

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ONBOARD_POLICY = 8  # Value of the firmware's stabilizer.controller parameter that runs the actor


def main():
    """Fly the configured trajectory with the onboard policy."""
    config = toml.load(Path(__file__).parent / "config.toml")

    ctrl_type = config["controller"]["type"]
    ctrl_freq = config["controller"]["ctrl_freq"]
    ctrl_interface = control_interface(ctrl_type, onboard=True)
    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    pos_start, yaw_start = trajectory_config.start_pose(trajectory)

    model = config["drone"]["model"]
    if not model.startswith("cf"):
        raise NotImplementedError(f"The {ctrl_type} policy is not deployed on {model}")

    initial_obs = {
        "pos": np.zeros(3),
        "quat": np.zeros(4),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "rpy": np.zeros(3),
        "t": time.perf_counter(),
    }
    # The firmware runs the network and takes the reference state as its goal
    controller = StateController(initial_obs, trajectory)

    drone = connect(config, ctrl_interface)
    # Reduce estimation noise for more accurate tracking
    drone.cf.param.set_value("locSrv.extPosStdDev", 0.002)
    try:
        takeoff(drone, pos_start, yaw_start)
        drone.cf.param.set_value("stabilizer.controller", ONBOARD_POLICY)
        data = track(drone, controller, ctrl_freq, trajectory_config.start_time)
        logger.info("Storing data...")
        path = new_recording_path(RECORDINGS_DIR, generate_filename_from_config(config, True))
        with open(path, "wb") as f:
            pickle.dump(data, f)
        drone.return_to_start()
    finally:
        drone.close()


if __name__ == "__main__":
    main()
