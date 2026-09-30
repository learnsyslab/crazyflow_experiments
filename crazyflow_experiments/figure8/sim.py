"""Fly a figure-8 policy, the MPC or the Mellinger controller in the Crazyflow simulator.

Run with
    pixi run -e figure8 python crazyflow_experiments/figure8/sim.py [--dr_scale 0.2]
"""

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import logging
import pickle
from pathlib import Path

import fire
import numpy as np
import toml
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line

from crazyflow_experiments.control.config import SOFT_START_TIME, TrajectoryConfig
from crazyflow_experiments.control.interface import control_interface
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.control.model_predictive_controller import ModelPredictiveController
from crazyflow_experiments.control.simulator import extract_obs
from crazyflow_experiments.control.takeoff import takeoff
from crazyflow_experiments.figure8 import RECORDINGS_DIR
from crazyflow_experiments.figure8.controller import AttitudeRL
from crazyflow_experiments.sim2real import ATTITUDE_FREQ, PHYSICS_FREQ
from crazyflow_experiments.sim2real.utils import generate_filename_from_config
from crazyflow_experiments.utils.paths import new_recording_path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(dr_scale: float = 0.0, config: dict | None = None) -> None:
    """Simulate the configured flight and save its recording.

    Args:
        dr_scale: Domain randomization scale of the policy checkpoint, 0 for the nominal one.
        config: Experiment config, `config.toml` next to this file if None.
    """
    if config is None:
        config = toml.load(Path(__file__).parent / "config.toml")
    data = []

    # Set up initial info
    ctrl_type = config["controller"]["type"]
    ctrl_freq = config["controller"]["ctrl_freq"]
    model = config["drone"]["model"]
    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    pos_start, yaw_start = trajectory_config.start_pose(trajectory)
    # get reference for rendering
    waypoints_time = np.arange(0, trajectory.cycle_time, 0.1)
    pos_reference = np.asarray(trajectory.pos(waypoints_time))

    # Set up sim
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone=model,
        dynamics=config["sim"]["physics"],
        control=control_interface(ctrl_type),
        freq=PHYSICS_FREQ,
        attitude_freq=ATTITUDE_FREQ,
        device="cpu",
    )
    sim.reset()

    # Prepare the controller (not necessary for sim, but for consistency with real)
    initial_obs = extract_obs(sim)
    match ctrl_type:
        case "mellinger":
            controller = MellingerController(initial_obs, trajectory, model)
        case "ppo" | "bptt":
            controller = AttitudeRL(
                initial_obs,
                trajectory,
                ctrl_type,
                model,
                ctrl_freq,
                config["experiment"]["time"],
                dr_scale,
            )
        case "mpc":
            controller = ModelPredictiveController(initial_obs, trajectory, model, ctrl_freq)
        case _:
            raise ValueError(f"Unknown controller type {ctrl_type}: mellinger, mpc, ppo, bptt")

    takeoff(sim, pos_start, yaw_start, ctrl_freq, render=config["sim"]["render"])

    # Experiment
    initial_obs = extract_obs(sim)
    controller._t_start = initial_obs["t"]  # Reset start time
    if trajectory_config.soft_start:
        controller._t_start += SOFT_START_TIME
    controller_finished = False
    i = 0
    while not controller_finished:
        obs = extract_obs(sim)
        action = controller(obs)
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
            sim.render()
        i += 1
    sim.close()

    config["controller"]["type"] += f"_dr{float(dr_scale)}" if dr_scale > 0 else ""
    path = new_recording_path(RECORDINGS_DIR, generate_filename_from_config(config))
    logger.info(f"Storing data to {path}")
    with open(path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    fire.Fire(main)
