"""Fly the system identification trajectories on the real drone.

The trajectory set named in `config.toml` is flown one after another around the origin, with a
Crazyflie over the radio or a PX4 vehicle through MAVROS depending on the drone model, like
`sim2real/deploy.py`. The sets end with a zero-size trajectory, so the flight ends with a hover.

The flight itself is the data. For the abstracted model, a Crazyflie needs a rosbag recorded
alongside (see the README) and a PX4 vehicle its ulog, which `data_converter_mocap.py` and
`data_converter_px4.py` turn into the csvs that `fit_simplified_model.py` fits. For the low-level
identification, a Crazyflie streams the firmware variables of `log_variables`, which are resampled
to the control frequency and stored as csv for `lol_sysid.py`.

Usage:

    pixi run -e sysid python crazyflow_experiments/sysid/collect_data.py

"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import toml

from crazyflow_experiments.control.config import TrajectorySetConfig
from crazyflow_experiments.control.interface import control_interface
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.control.state_controller import StateController
from crazyflow_experiments.deploy.experiment import connect, finish, start, track
from crazyflow_experiments.sysid import RECORDINGS_DIR
from crazyflow_experiments.utils.paths import new_recording_path

if TYPE_CHECKING:
    from cflib.crazyflie.log import LogConfig
    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def resample(
    samples: list[tuple[float, dict[str, float]]], variables: list[str], freq: float
) -> NDArray[np.floating]:
    """Resample the streamed firmware variables onto a common time grid.

    The firmware streams the variables in blocks with their own timestamps, so every variable is
    interpolated linearly from its own samples onto a grid over the time span all of them cover.

    Args:
        samples: Firmware timestamp in ms and the variables of one block, as they were streamed.
        variables: Variables in the order of the returned columns.
        freq: Grid frequency in Hz.

    Returns:
        Rows of the time in ms followed by the variables, shape (N, len(variables) + 1).
    """
    series = {name: np.array([(t, d[name]) for t, d in samples if name in d]) for name in variables}
    t_start = max(s[0, 0] for s in series.values())
    t_end = min(s[-1, 0] for s in series.values())
    t = np.arange(t_start, t_end, 1000 / freq)
    return np.column_stack((t, *(np.interp(t, s[:, 0], s[:, 1]) for s in series.values())))


def main(calibrate: bool = True):
    """Fly the trajectory set with the configured controller."""
    config = toml.load(Path(__file__).parent / "config.toml")

    ctrl_type = config["controller"]["type"]
    ctrl_freq = config["controller"]["ctrl_freq"]
    ctrl_interface = control_interface(ctrl_type, onboard=True)
    set_config = TrajectorySetConfig.from_config(config)
    trajectory = set_config.build()
    pos_start, yaw_start = set_config.start_pose(trajectory)
    logger.info(f"Flying {set_config.trajectories}, {trajectory.duration:.0f}s from {pos_start}")

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
                legacy_heading=True,  # The identified models expect the firmware's attitude
            )
        case "state":
            controller = StateController(initial_obs, trajectory)
        case _:
            raise ValueError(f"Unknown controller type: {ctrl_type}")

    # A Crazyflie streams the firmware variables in blocks, collected as they arrive
    log_variables = config["experiment"]["log_variables"]
    samples: list[tuple[float, dict[str, float]]] = []

    def log_onboard(timestamp: float, data: dict[str, float], logconf: LogConfig):
        samples.append((timestamp, data))

    drone = connect(config, ctrl_interface, log_variables or None, log_onboard)
    try:
        start(drone, pos_start, yaw_start, ctrl_freq, calibrate=calibrate)
        track(drone, controller, ctrl_freq, set_config.start_time)
        if samples:
            path = new_recording_path(RECORDINGS_DIR, f"lol_sysid_{model}", ".csv")
            header = "time," + ",".join(log_variables)
            np.savetxt(
                path, resample(samples, log_variables, ctrl_freq), ",", header=header, comments=""
            )
            logger.info(f"Stored the firmware variables resampled to {ctrl_freq} Hz in {path}")
        finish(drone)
    finally:
        drone.close()


if __name__ == "__main__":
    main()
