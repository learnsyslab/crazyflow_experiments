"""Fly the reach-position policy in the Crazyflow simulator and record the flight.

The policy commands rotor velocities and chases the current point of the trajectory. There is no
takeoff: the drone starts hovering below the trajectory origin and the policy flies to the start.

Run with
    pixi run -e learn2fly python crazyflow_experiments/learn2fly/sim.py
"""

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import logging
import pickle
from pathlib import Path

import jax.numpy as jp
import numpy as np
import toml
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line
from crazyflow.utils import leaf_replace

from crazyflow_experiments.control.config import SOFT_START_TIME, TrajectoryConfig
from crazyflow_experiments.control.interface import control_interface
from crazyflow_experiments.control.simulator import extract_obs
from crazyflow_experiments.core.envs.drone_env import HOVER_RPM
from crazyflow_experiments.learn2fly import RECORDINGS_DIR
from crazyflow_experiments.learn2fly.controller import Learn2FlyRL
from crazyflow_experiments.sim2real import ATTITUDE_FREQ, PHYSICS_FREQ
from crazyflow_experiments.sim2real.utils import generate_filename_from_config
from crazyflow_experiments.utils.paths import new_recording_path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

START_BELOW_ORIGIN = 0.3  # m the drone starts below the trajectory origin


def main(config: dict | None = None) -> None:
    """Simulate the configured flight and save its recording.

    Args:
        config: Experiment config, `config.toml` next to this file if None.
    """
    if config is None:
        config = toml.load(Path(__file__).parent / "config.toml")
    data = []

    # Set up initial info
    ctrl_type = config["controller"]["type"]
    ctrl_freq = config["controller"]["ctrl_freq"]
    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    # get reference for rendering
    waypoints_time = np.arange(0, trajectory.cycle_time, 0.1)
    pos_reference = np.asarray(trajectory.pos(waypoints_time))

    # Set up sim
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone=config["drone"]["model"],
        dynamics=config["sim"]["physics"],
        control=control_interface(ctrl_type),
        freq=PHYSICS_FREQ,
        attitude_freq=ATTITUDE_FREQ,
        device="cpu",
    )
    sim.reset()

    # No takeoff for the policy. We have no rotor vel-level takeoff controller
    pos_start = jp.array(trajectory_config.origin, dtype=jp.float32).at[2].add(-START_BELOW_ORIGIN)
    pos_start = pos_start[None, None, :]  # (1, 1, 3)
    hover_rpm = jp.full((1, 1, 4), HOVER_RPM)
    sim.data = sim.data.replace(
        states=leaf_replace(sim.data.states, None, pos=pos_start, rotor_vel=hover_rpm),
        core=sim.data.core.replace(mjx_synced=False),
    )
    sim.rotor_vel_control(hover_rpm)

    # Prepare the controller
    initial_obs = extract_obs(sim)
    controller = Learn2FlyRL(initial_obs, trajectory)

    # Experiment
    controller._t_start = initial_obs["t"]  # Reset start time
    if trajectory_config.soft_start:
        controller._t_start += SOFT_START_TIME
    controller_finished = False
    i = 0
    while not controller_finished:
        obs = extract_obs(sim)
        action = controller(obs)
        sim.rotor_vel_control(action[None, None, :])
        sim.step(sim.freq // ctrl_freq)

        obs["u"] = action
        data.append(obs | {"t": obs["t"] - controller._t_start})  # Recorded in trajectory time

        controller_finished = controller.done(obs)
        if controller_finished:
            break

        if (
            config["sim"]["render"]
            and ((i * config["sim"]["fps"]) % ctrl_freq) < config["sim"]["fps"]
        ):
            draw_line(sim, pos_reference, [0, 1, 0, 1])
            times_future = obs["t"] - controller._t_start + np.arange(0.0, 1.0, 0.1)
            pos_reference_future = np.asarray(trajectory.pos(times_future))
            draw_line(sim, pos_reference_future, [1, 0, 0, 1], start_size=10, end_size=10)
            sim.render()
        i += 1
    sim.close()

    path = new_recording_path(RECORDINGS_DIR, generate_filename_from_config(config))
    logger.info(f"Storing data to {path}")
    with open(path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
