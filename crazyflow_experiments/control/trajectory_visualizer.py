"""Plot the trajectory an experiment config describes.

Usage:

    pixi run -e sim2real python crazyflow_experiments/control/trajectory_visualizer.py <config.toml>

e.g. with `crazyflow_experiments/sim2real/config.toml` or `crazyflow_experiments/sysid/config.toml`.
The config is read like the flight scripts read it: `trajectory` in the experiment section gives a
single periodic curve, `trajectories` a set flown one after another. The path is drawn in 3D and
colored by time, next to the position and heading over time with the transitions shaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import toml

from crazyflow_experiments.control.config import FlightStart, TrajectoryConfig, TrajectorySetConfig
from crazyflow_experiments.control.trajectories import CompositeTrajectory

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from crazyflow_experiments.control.trajectories import Trajectory


def load_config(path: Path) -> FlightStart:
    """Read the trajectory or trajectory set configuration of a config file."""
    config = toml.load(path)
    if "trajectories" in config["experiment"]:
        return TrajectorySetConfig.from_config(config)
    return TrajectoryConfig.from_config(config)


def plot_trajectory(trajectory: Trajectory, start_time: float, freq: float = 50.0) -> Figure:
    """Draw the path in 3D and the coordinates over time.

    Args:
        trajectory: Trajectory to draw.
        start_time: Time the drone joins the trajectory, negative for a soft start.
        freq: Samples per second.
    """
    t = np.arange(start_time, trajectory.duration, 1 / freq)
    pos = np.asarray(trajectory.pos(t))
    yaw = np.asarray(trajectory.yaw(t))
    speed = np.linalg.norm(np.asarray(trajectory.vel(t)), axis=1)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    points = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=t, s=2)
    ax.set_box_aspect(np.ptp(pos, axis=0) + 1e-3)  # equal scale on all axes
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_zlabel("$z$ [m]")
    fig.colorbar(points, ax=ax, label="$t$ [s]", shrink=0.6)

    axes = [fig.add_subplot(3, 2, 2 * i + 2) for i in range(3)]
    axes[0].plot(t, pos, label=["$x$", "$y$", "$z$"])
    axes[0].set_ylabel("position [m]")
    axes[1].plot(t, speed)
    axes[1].set_ylabel("speed [m/s]")
    axes[2].plot(t, yaw)
    axes[2].set_ylabel("yaw [rad]")
    axes[2].set_xlabel("$t$ [s]")
    axes[0].legend(loc="upper right", ncols=3)
    if isinstance(trajectory, CompositeTrajectory):
        for start, part, transition in zip(
            trajectory.starts, trajectory.trajectories, (*trajectory.transitions, 0.0), strict=True
        ):
            for axis in axes:
                axis.axvspan(start + part.duration, start + part.duration + transition, alpha=0.2)
    for axis in axes:
        axis.grid(True)
    fig.tight_layout()
    return fig


def main():
    """Plot the trajectory of the config given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="config file with an experiment section")
    args = parser.parse_args()
    config = load_config(args.config)
    plot_trajectory(config.build(), config.start_time)
    plt.show()


if __name__ == "__main__":
    main()
