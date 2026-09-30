"""Deploy the MPPI obstacle avoidance controller on the real drone.

Run with
    pixi run -e mppi python crazyflow_experiments/mppi_obstacles/deploy.py
    pixi run -e mppi python crazyflow_experiments/mppi_obstacles/deploy.py --calibrate False
"""

import logging
import pickle
import time
from pathlib import Path

import fire
import numpy as np
import rclpy

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.deploy.experiment import connect
from crazyflow_experiments.deploy.takeoff import takeoff
from crazyflow_experiments.mppi_obstacles import RECORDINGS_DIR
from crazyflow_experiments.mppi_obstacles.controller import AttitudeMPPIController
from crazyflow_experiments.mppi_obstacles.sim import build_initial_info
from crazyflow_experiments.mppi_obstacles.utils import experiment_name, load_config
from crazyflow_experiments.utils.paths import new_recording_path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(calibrate: bool = True, save: bool = True):
    """Main deployment loop.

    Args:
        calibrate: Calibrate the drone orientation once it hovers after takeoff.
        save: Write the recorded observations and actions to the data folder.
    """
    folder = Path(__file__).parent
    config = load_config(folder)

    data = []

    initial_info, trajectory = build_initial_info(config)
    pos_start, yaw_start = TrajectoryConfig.from_config(config).start_pose(trajectory)

    # Prepare controllers
    initial_obs = {
        "pos": np.zeros(3),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "rpy": np.zeros(3),
        "is_outdated": False,
        "pos_raw": np.zeros(3),
        "quat_raw": np.array([0.0, 0.0, 0.0, 1.0]),
        "t": 0.0,
    }

    controller = AttitudeMPPIController(
        initial_obs, initial_info, trajectory, TrajectoryConfig.from_config(config).soft_start
    )

    # Set up deploy env (Needs to be here, otherwise drone times out)
    model = config["drone"]["model"]
    if "cf" not in model:
        raise NotImplementedError(f"Deployment for {model} not implemented")
    drone = connect(config, initial_info["ctrl_interface"])

    try:
        takeoff(drone, pos_start, yaw_start, calibrate=calibrate)

        # region Experiment
        controller.start(time.perf_counter())
        first_obs = drone.obs()
        first_obs["obstacles"] = config["experiment"]["env"]["obstacles"]
        data.append(first_obs.copy())
        while rclpy.ok():
            t = time.perf_counter()
            obs = drone.obs()
            obs["t"] = t

            if obs["is_outdated"]:
                logger.info("Measurements outdated. Emergency stop.")
                break

            t0 = time.perf_counter()
            action = controller(obs)
            t1 = time.perf_counter()
            drone.send_action(action)

            obs["u"] = action
            obs["compute_time"] = t1 - t0
            data.append(obs.copy())

            controller_finished = controller.done(obs)
            if controller_finished:
                logger.info("Controller terminated. Returning home.")
                break

            drone.update_estimators()

            sleep_until_next_cycle(t, initial_info["ctrl_freq"])

        drone.return_to_start()
        if save:
            path = new_recording_path(RECORDINGS_DIR, experiment_name(config, "mppi", "deploy"))
            logger.info(f"Storing data to {path}")
            with open(path, "wb") as f:
                pickle.dump(data, f)

    finally:  # Kill the drone
        drone.close()


if __name__ == "__main__":
    fire.Fire(main)
