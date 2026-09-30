"""This file is used to run the controllers in simulation.

Run with
    pixi run -e sim2real python crazyflow_experiments/sim2real/sim_crazyflow.py
"""

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import logging
import pickle
from pathlib import Path

import numpy as np
import toml
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line, draw_points

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.interface import control_interface
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.control.simulator import extract_obs
from crazyflow_experiments.control.state_controller import StateController
from crazyflow_experiments.control.takeoff import takeoff
from crazyflow_experiments.sim2real import ATTITUDE_FREQ, PHYSICS_FREQ, RECORDINGS_DIR
from crazyflow_experiments.sim2real.utils import next_recording_path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def rollout(config: dict) -> None:
    """Simulate one rollout of the experiment and save its recording."""
    data = []

    # Set up initial info
    ctrl_freq = config["controller"]["ctrl_freq"]
    ctrl_interface = control_interface(config["controller"]["type"])
    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    pos_start, yaw_start = trajectory_config.start_pose(trajectory)
    # get reference for rendering
    waypoints_time = np.arange(0, trajectory.cycle_time, 0.1)
    pos_reference = np.asarray(trajectory.pos(waypoints_time))

    # Set up sim
    ctrl_type = config["controller"]["type"]
    sim_control = ctrl_interface
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone=config["drone"]["model"],
        dynamics=config["sim"]["physics"],
        control=sim_control,
        freq=PHYSICS_FREQ,
        attitude_freq=ATTITUDE_FREQ,
        device="cpu",
    )
    sim.reset()

    # Prepare controllers (not necessary for sim, but for consistency with real)
    initial_obs = extract_obs(sim)
    match config["controller"]["type"]:
        case "mellinger":
            controller = MellingerController(
                initial_obs,
                trajectory,
                config["drone"]["model"],
                tilt_max=np.deg2rad(config["controller"]["tilt_max"]) or None,
            )
        case "state":
            controller = StateController(initial_obs, trajectory)
        case _:
            raise ValueError(f"Unknown controller type: {config['controller']['type']}")

    takeoff(sim, pos_start, yaw_start, ctrl_freq, render=config["sim"]["render"])

    # Experiment
    # Actual experiment controller
    initial_obs = extract_obs(sim)
    controller._t_start = initial_obs["t"]  # Reset start time
    controller._t_start -= trajectory_config.start_time  # Negative for a soft start
    controller_finished = False
    i = 0
    while not controller_finished:
        obs = extract_obs(sim)
        action = controller(obs)
        if ctrl_type == "state":
            sim.state_control(action[None, None, :])
        else:
            sim.attitude_control(action[None, None, :])
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
            if ctrl_type == "state":
                draw_points(sim, action[None, 0:3], [1, 1, 1, 1])
                vel_vec = np.array([action[0:3], action[0:3] + action[3:6] * 0.1])
                acc_vec = np.array([action[0:3], action[0:3] + action[6:9] * 0.1])
                draw_line(sim, vel_vec, [0, 0, 1, 1])
                draw_line(sim, acc_vec, [1, 0, 0, 1])
            sim.render()
        i += 1
    sim.close()
    with open(next_recording_path(RECORDINGS_DIR, config), "wb") as f:
        pickle.dump(data, f)


def main(config: dict | None = None) -> None:
    """Repeat the rollout `n_rollouts` times, each saved under the next free index."""
    if config is None:
        config = toml.load(Path(__file__).parent / "config.toml")
    for _ in range(config["sim"]["n_rollouts"]):
        rollout(config)


if __name__ == "__main__":
    main()
