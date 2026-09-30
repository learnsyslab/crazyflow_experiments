"""Deploy the Crazyflie with manual (gamepad) control."""

import logging
import os
import pickle
import time
from pathlib import Path

import rclpy
import toml

from crazyflow_experiments.control.interface import control_interface
from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.deploy.experiment import connect
from crazyflow_experiments.sim2real.utils import generate_filename_from_config
from crazyflow_experiments.utils.paths import RESULTS_DIR
from crazyflow_experiments.vision.control.gamepad_controller import GamepadController
from crazyflow_experiments.vision.ros.imu_publisher import ImuPublisher

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    """Main deployment loop."""
    folder = Path(__file__).parent
    config = toml.load(folder / "config.toml")

    data = []

    # Set up initial info
    ctrl_freq = config["controller"]["ctrl_freq"]
    ctrl_interface = control_interface(config["controller"]["type"], onboard=True)
    if not config["controller"]["type"] == "gamepad":
        raise ValueError(f"Controller must be 'gamepad', got {config['controller']['type']}")
    controller = GamepadController(config["drone"]["model"], ctrl_freq)

    # Deploy reuses this ROS context. The bound method has the callback
    # signature expected by cflib's LogConfig.
    rclpy.init()
    drone_name = f"cf{config['drone']['id']}"
    imu_publisher = ImuPublisher(drone_name)
    log_variables = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]

    # Set up deploy env
    model = config["drone"]["model"]
    if "cf" not in model:
        raise NotImplementedError(f"Deployment for {model} not implemented")
    drone = connect(config, ctrl_interface, log_variables, imu_publisher.cf_log_callback)
    try:
        # Experiment
        while rclpy.ok():
            t = time.perf_counter()
            obs = drone.obs()
            obs["t"] = t

            if obs["is_outdated"]:
                logger.info("Measurements outdated. Emergency stop.")
                break

            action = controller(obs)
            drone.send_action(action)

            obs["u"] = action
            data.append(obs.copy())

            controller_finished = controller.done(obs)
            if controller_finished:
                logger.info("Controller terminated. Returning home.")
                break

            if "state" in config["controller"]["type"]:
                drone.update_estimators()

            sleep_until_next_cycle(t, ctrl_freq)

        counter = 0
        logger.info("Storing data...")
        while (
            filepath := (
                RESULTS_DIR
                / "vision"
                / f"{generate_filename_from_config(config, True)}_{counter:02d}.pkl"
            )
        ).exists():
            counter += 1
            if counter > 99:
                raise FileExistsError("Too many files with the same name exist already.")

        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        drone.return_to_start()

    finally:  # Kill the drone
        try:
            drone.close()
        finally:
            imu_publisher.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
