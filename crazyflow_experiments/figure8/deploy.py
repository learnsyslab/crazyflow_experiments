"""This file takes the deploy arguments and runs the platform specific deploy code."""

import logging
import os
import pickle
import time
from pathlib import Path

import fire
import numpy as np
import toml

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.interface import control_interface
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.control.model_predictive_controller import ModelPredictiveController
from crazyflow_experiments.deploy.experiment import connect, track
from crazyflow_experiments.deploy.takeoff import takeoff
from crazyflow_experiments.figure8 import RECORDINGS_DIR
from crazyflow_experiments.figure8.controller import AttitudeRL
from crazyflow_experiments.sim2real.utils import generate_filename_from_config
from crazyflow_experiments.utils.paths import new_recording_path

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

logger = logging.getLogger(__name__)


def main(calibrate: bool = True, save: bool = True, dr_scale: float = 0.0):
    """Main deployment loop.

    Recorded observations carry the trajectory time under "t".

    Args:
        calibrate: Calibrate the drone orientation once it hovers after takeoff.
        save: Write the recorded observations and actions to the data folder.
        dr_scale: Domain randomization scale of the policy checkpoint, 0 for the nominal one.
    """
    folder = Path(__file__).parent
    config = toml.load(folder / "config.toml")
    model = config["drone"]["model"]

    # Set up initial info
    ctrl_freq = config["controller"]["ctrl_freq"]
    ctrl_interface = control_interface(config["controller"]["type"])
    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    pos_start, yaw_start = trajectory_config.start_pose(trajectory)

    # Prepare controllers
    initial_obs = {
        "pos": np.zeros(3),
        "quat": np.zeros(4),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "rpy": np.zeros(3),
        "t": time.perf_counter(),
    }
    if config["controller"]["type"] == "mellinger":
        controller_rl = MellingerController(initial_obs, trajectory, model)
    elif config["controller"]["type"] == "mpc":
        controller_rl = ModelPredictiveController(initial_obs, trajectory, model, ctrl_freq)
    else:
        controller_rl = AttitudeRL(
            initial_obs,
            trajectory,
            config["controller"]["type"],
            "cf21B_500" if model == "cf21B_350" else model,
            ctrl_freq,
            config["experiment"]["time"],
            dr_scale,
        )

    # Set up deploy env
    if "cf" not in model:
        raise NotImplementedError(f"Deployment for {model} not implemented")
    drone = connect(config, ctrl_interface)

    try:
        takeoff(drone, pos_start, yaw_start, calibrate=calibrate)
        data = track(drone, controller_rl, ctrl_freq, trajectory_config.start_time)
        drone.return_to_start()
    finally:  # Kill the drone
        drone.close()

    if save:
        config["controller"]["type"] += f"_dr{float(dr_scale)}" if dr_scale > 0 else ""
        name = generate_filename_from_config(config, True)
        path = new_recording_path(RECORDINGS_DIR, name)
        logger.info(f"Storing data to {path}")
        with open(path, "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
