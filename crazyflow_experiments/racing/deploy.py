"""Deploy racing on real hardware.

Run with
    pixi run -e racing python crazyflow_experiments/racing/deploy.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np
import rclpy
from crazyflow.dynamics import Dynamics
from lsy_drone_racing.utils import load_config, load_controller

from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.utils.paths import RESULTS_DIR

if TYPE_CHECKING:
    from lsy_drone_racing.envs.real_race_env import RealDroneRaceEnv

logger = logging.getLogger(__name__)


def main(config: str = "track.toml", controller: str | None = None):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `racing/control/` or None. If None,
            the controller specified in the config file is used.
    """
    rclpy.init()
    config = load_config(Path(__file__).parents[0] / "config" / config)
    env: RealDroneRaceEnv = gymnasium.make(
        "RealDroneRacing-v0",
        drones=config.deploy.drones,
        freq=config.env.freq,
        track=config.env.track,
        randomizations=config.env.randomizations,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        dynamics=Dynamics(config.sim.dynamics),
    )
    try:
        obs, info = env.reset(seed=config.env.seed, options=config.deploy)
        next_obs = obs  # Set next_obs to avoid errors when the loop never enters
        obs_list = []

        control_path = Path(__file__).parents[0] / "control"
        controller_path = control_path / (controller or config.controller.file)
        controller_cls = load_controller(controller_path)
        controller = controller_cls(obs, info, config)
        start_time = time.perf_counter()
        while rclpy.ok():
            t_loop = time.perf_counter()
            obs, info = env.unwrapped.obs(), env.unwrapped.info()
            obs = {k: v[0] for k, v in obs.items()}
            action = controller.compute_control(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(
                action, next_obs, reward, terminated, truncated, info
            )
            obs_list.append(
                [time.time(), *action, *obs["pos"], *obs["quat"], *obs["vel"], *obs["ang_vel"]]
            )
            if terminated or truncated or controller_finished:
                break
            sleep_until_next_cycle(t_loop, config.env.freq)
        ep_time = time.perf_counter() - start_time
        finished_track = next_obs["n_gates_passed"] == next_obs["gate_sequence"].shape[0]
        logger.info(f"Track time: {ep_time:.3f}s" if finished_track else "Task not completed")
        obs_list = np.array(obs_list)
        path = RESULTS_DIR / "racing" / "obs_deploy.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "time,cmd_roll,cmd_pitch,cmd_yaw,cmd_thrust,"
            "pos_x,pos_y,pos_z,"
            "quat_x,quat_y,quat_z,quat_w,"
            "vel_x,vel_y,vel_z,"
            "ang_vel_x,ang_vel_y,ang_vel_z"
        )
        np.savetxt(path, obs_list, delimiter=",", header=header, comments="")
        logger.info(f"Saved observations to {path}")
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    fire.Fire(main)
