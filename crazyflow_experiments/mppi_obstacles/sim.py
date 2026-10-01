"""Run MPPI obstacle avoidance controllers in simulation.

Run with
    pixi run -e mppi python crazyflow_experiments/mppi_obstacles/sim.py
    pixi run -e mppi python crazyflow_experiments/mppi_obstacles/sim.py --render False
"""

import os

# Both must be set before scipy and jax are imported
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Any

import fire
import h5py
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_capsule, draw_line

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.simulator import extract_obs
from crazyflow_experiments.control.takeoff import takeoff
from crazyflow_experiments.control.trajectories import Trajectory
from crazyflow_experiments.mppi_obstacles import RECORDINGS_DIR
from crazyflow_experiments.mppi_obstacles.controller import AttitudeMPPIController
from crazyflow_experiments.mppi_obstacles.utils import experiment_name, load_config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SIM_FREQ = 1000  # Hz, physics integration frequency
ATTITUDE_FREQ = 500  # Hz, onboard attitude controller frequency


def build_initial_info(config: dict) -> tuple[dict[str, Any], Trajectory]:
    """Build the initial_info dict and the trajectory from config."""
    experiment = config["experiment"]
    info: dict[str, Any] = {
        "drone_model": config["drone"]["model"],
        "ctrl_freq": config["controller"]["ctrl_freq"],
        "ctrl_interface": Control.attitude,
        **experiment["env"],
        "controller": config["controller"],
    }
    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    return info, trajectory


def create_sim(config: dict) -> Sim:
    """Instantiate and reset the Crazyflow simulator."""
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone=config["drone"]["model"],
        dynamics=config["sim"]["physics"],
        control=Control.attitude,
        freq=SIM_FREQ,
        attitude_freq=ATTITUDE_FREQ,
        device="cpu",
    )
    sim.reset()
    return sim


def render_obstacles(sim: Sim, initial_info: dict[str, Any]):
    """Draw all obstacle capsules into the scene."""
    for o in np.array(initial_info["obstacles"]):
        draw_capsule(
            sim,
            o,
            o + np.array([0, 0, initial_info["obstacle_length"]]),
            radius=initial_info["obstacle_radius"],
            rgba=np.array([1.0, 1.0, 1.0, 0.5]),
            cylinder=True,
        )


def should_render(step: int, fps: int, ctrl_freq: int) -> bool:
    """Check whether the current step should trigger a render frame."""
    return ((step * fps) % ctrl_freq) < fps


def run_experiment(
    sim: Sim,
    config: dict,
    controller: AttitudeMPPIController,
    initial_info: dict[str, Any],
    record: bool,
    render: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]] | None]:
    """Run the main experiment loop and return collected data and recording."""
    ctrl_freq = config["controller"]["ctrl_freq"]
    fps = config["sim"]["fps"]
    mppi_cfg = initial_info["controller"]["mppi"]

    controller.start(sim.data.core.steps[0, 0] / sim.freq)

    rgbas = np.random.default_rng(0).uniform(0, 1, (mppi_cfg["K"], 4))
    rgbas[..., 3] = 1.0
    pos_hist: deque[Any] = deque(maxlen=100)
    data: list[dict[str, Any]] = []
    recording: dict[str, list[np.ndarray]] = {
        "pos": [],
        "quat": [],
        "reference": [],
        "mppi_rollouts": [],
    }

    for step in range(10_000_000):
        obs = extract_obs(sim)
        pos_hist.append(sim.data.states.pos[0, 0])

        t_control = time.perf_counter()
        action = controller(obs)
        obs["compute_time"] = time.perf_counter() - t_control
        sim.attitude_control(action[None, None, :])
        sim.step(sim.freq // ctrl_freq)

        obs["u"] = action
        data.append(obs.copy())

        if record:
            recording["pos"].append(np.asarray(obs["pos"], dtype=np.float32))
            recording["quat"].append(np.asarray(obs["quat"], dtype=np.float32))
            recording["reference"].append(controller.reference_positions(obs["t"]))
            pos = np.asarray(controller.positions.mean(axis=1), dtype=np.float32)
            recording["mppi_rollouts"].append(pos)

        if controller.done(obs):
            break

        if render and should_render(step, fps, ctrl_freq):
            _render_experiment_frame(sim, obs, controller, initial_info, pos_hist, rgbas)
    else:
        raise RuntimeError("Experiment did not finish")

    return data, recording if record else None


def _render_experiment_frame(
    sim: Sim,
    obs: dict[str, Any],
    controller: AttitudeMPPIController,
    initial_info: dict[str, Any],
    pos_hist: deque,
    rgbas: np.ndarray,
):
    """Render one experiment frame: rollouts, reference, history, and obstacles."""
    pos = np.asarray(controller.positions.mean(axis=1), dtype=np.float32)
    for k in range(controller.K):
        draw_line(sim, pos[k], rgbas[k], start_size=0.1, end_size=0.1)

    des_pos = controller.reference_positions(obs["t"])
    draw_line(sim, des_pos, np.array([0.0, 0.25, 1.0, 1.0]), start_size=0.3, end_size=3.0)

    # Position history
    draw_line(
        sim, np.array(pos_hist), np.array([1.0, 0.25, 0.0, 1.0]), start_size=0.3, end_size=3.0
    )

    render_obstacles(sim, initial_info)
    sim.render()


def save_data(data: list[dict[str, Any]], config: dict):
    """Pickle experiment data to the recordings directory."""
    name = experiment_name(config, "mppi", config["sim"]["physics"])
    path = RECORDINGS_DIR / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def save_recording(
    recording: dict[str, list[np.ndarray]], config: dict, initial_info: dict[str, Any]
):
    """Write recorded trajectory data and metadata to an H5 file."""
    mppi_cfg = initial_info["controller"]["mppi"]
    path = RECORDINGS_DIR / "crazyflow_mppi.h5"
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Timeseries datasets
        f.create_dataset("pose/pos", data=np.stack(recording["pos"]))
        f.create_dataset("pose/quat", data=np.stack(recording["quat"]))
        f.create_dataset("reference/pos", data=np.stack(recording["reference"]))
        if recording["mppi_rollouts"]:
            f.create_dataset("mppi/rollouts", data=np.stack(recording["mppi_rollouts"]))

        # Metadata
        f.attrs["sim_freq"] = SIM_FREQ
        f.attrs["ctrl_freq"] = config["controller"]["ctrl_freq"]
        f.attrs["attitude_freq"] = ATTITUDE_FREQ
        f.attrs["drone_model"] = config["drone"]["model"]
        f.attrs["physics"] = config["sim"]["physics"]
        f.attrs["trajectory_type"] = config["experiment"]["trajectory"]
        f.attrs["trajectory_time"] = config["experiment"]["time"]
        f.attrs["scaling"] = config["experiment"]["scaling"]
        f.attrs["soft_start"] = config["experiment"]["soft_start"]

        # MPPI-specific metadata
        f.attrs["mppi_freq"] = config["controller"]["ctrl_freq"]
        f.attrs["mppi_N"] = mppi_cfg["N"]
        f.attrs["mppi_T"] = mppi_cfg["T"]
        f.attrs["mppi_K"] = mppi_cfg["K"]
        f.attrs["mppi_dt"] = mppi_cfg["T"] / mppi_cfg["N"]
        f.attrs["mppi_n_samples"] = mppi_cfg["n_samples"]
        f.attrs["mppi_temperature"] = mppi_cfg["temperature"]

        # Obstacle geometry
        f.create_dataset("obstacles/positions", data=np.array(initial_info["obstacles"]))
        f.attrs["obstacle_radius"] = initial_info["obstacle_radius"]
        f.attrs["obstacle_length"] = initial_info["obstacle_length"]
        f.attrs["drone_radius"] = initial_info["drone_radius"]

    logger.info(f"Recording saved to {path}")


def run(config: dict, record: bool = False, render: bool = True, save: bool = True) -> None:
    """Fly one MPPI experiment in simulation."""
    initial_info, trajectory = build_initial_info(config)
    pos_start, yaw_start = TrajectoryConfig.from_config(config).start_pose(trajectory)
    sim = create_sim(config)
    initial_obs = extract_obs(sim)
    controller = AttitudeMPPIController(
        initial_obs, initial_info, trajectory, TrajectoryConfig.from_config(config).soft_start
    )

    takeoff(sim, pos_start, yaw_start, config["controller"]["ctrl_freq"])
    data, recording = run_experiment(sim, config, controller, initial_info, record, render)

    sim.close()
    if save:
        save_data(data, config)
    if recording is not None:
        save_recording(recording, config, initial_info)


def main(record: bool = False, render: bool = True, save: bool = True):
    """Run the MPPI obstacle avoidance in simulation.

    Args:
        record: Save the trajectories, references and MPPI rollouts to an H5 file.
        render: Open the MuJoCo viewer.
        save: Save the experiment data to a pickle file.
    """
    run(load_config(Path(__file__).parent), record=record, render=render, save=save)


if __name__ == "__main__":
    fire.Fire(main)
