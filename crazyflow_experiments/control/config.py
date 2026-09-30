"""Configuration shared by the experiments that fly a trajectory."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import toml

from crazyflow_experiments.control.trajectories import (
    SOFT_START_TIME,
    TRAJECTORIES,
    CompositeTrajectory,
    Trajectory,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Controllers that produce a collective thrust and an attitude setpoint
ATTITUDE_CONTROLLERS = ("mellinger", "mpc", "ppo", "bptt", "gamepad", "state_manual")

# The physical parameters of every drone model and the trajectory sets of the identification flights
# are shared by the experiments, so they live in their own files next to the sysid config
SYSID_DIR = Path(__file__).parents[1] / "sysid"
DRONES_CONFIG = SYSID_DIR / "config_drones.toml"
TRAJECTORIES_CONFIG = SYSID_DIR / "config_trajectories.toml"


def drone_params(model: str) -> dict:
    """Physical parameters of a drone model.

    Args:
        model: Name of the drone model, e.g. "cf2x_L250" or "hb_x500".

    Returns:
        The `[<model>]` table of `sysid/config_drones.toml` with the mass, pwm range, thrust curve
        and thrust limits of the model.
    """
    drones = toml.load(DRONES_CONFIG)
    if model not in drones:
        raise KeyError(f"No [{model}] table in {DRONES_CONFIG} with the parameters of the model")
    return drones[model]


def build_trajectory(
    trajectory: str,
    n_runs: int,
    time: float,
    scaling: list[float],
    origin: list[float],
    axes: str = "xyz",
    yaw_mode: str = "constant",
    yaw_setting: float = 0.0,
) -> Trajectory:
    """Build a periodic trajectory from the keys the config files use.

    Args:
        trajectory: Name of the curve in `TRAJECTORIES`.
        n_runs: Number of cycles.
        time: Duration of one cycle in s.
        scaling: Extent of the curve in m along its own axes.
        origin: Position the curve is centered on.
        axes: World axes the curve's own x, y and z are placed on, e.g. "xzy" for an upright curve.
        yaw_mode: How the heading evolves, see `PeriodicTrajectory.yaw`.
        yaw_setting: Parameter of the yaw mode.
    """
    return TRAJECTORIES[trajectory](
        scaling=jnp.array(scaling),
        cycle_time=time,
        duration=n_runs * time,
        origin=jnp.array(origin),
        axes=axes,
        yaw_mode=yaw_mode,
        yaw_setting=yaw_setting,
    )


class FlightStart:
    """Where a flight joins its trajectory, shared by the trajectory configurations."""

    soft_start: bool

    @property
    def start_time(self) -> float:
        """Time at which the drone joins the trajectory, negative for a soft start."""
        return -SOFT_START_TIME if self.soft_start else 0.0

    def start_pose(self, trajectory: Trajectory) -> tuple[NDArray[np.floating], float]:
        """Return the pose at which the drone joins the trajectory."""
        t = self.start_time
        return np.asarray(trajectory.pos(t)), float(trajectory.yaw(t))


@dataclass(frozen=True)
class TrajectoryConfig(FlightStart):
    """Trajectory the drone flies, as named in the experiment section of a config file."""

    trajectory: str
    n_runs: int
    origin: list[float]
    plane: str
    scaling: list[float]
    time: float
    yaw_mode: str
    yaw_setting: float
    soft_start: bool

    @classmethod
    def from_config(cls, config: dict) -> TrajectoryConfig:
        """Read the trajectory settings from the experiment section of a config."""
        experiment = config["experiment"]
        return cls(**{field.name: experiment[field.name] for field in fields(cls)})

    def build(self) -> Trajectory:
        """Build the trajectory, compiled for the flight."""
        trajectory = build_trajectory(
            self.trajectory,
            self.n_runs,
            self.time,
            self.scaling,
            self.origin,
            axes="xyz" if self.plane == "xy" else "xzy",
            yaw_mode=self.yaw_mode,
            yaw_setting=self.yaw_setting,
        )
        trajectory.warm_up(self.start_time)
        return trajectory


@dataclass(frozen=True)
class TrajectorySetConfig(FlightStart):
    """Trajectories flown one after another, as named in the experiment section of a config file.

    The experiment section names the set with `trajectories` and gives the common `origin`, the
    `transition_time` between trajectories that do not connect, and `soft_start`. The set is a list
    of trajectory tables under `[trajectory_sets]` of `sysid/config_trajectories.toml`, or of the
    config itself if it has the table, each with the keys of `build_trajectory` except the origin.
    """

    trajectories: str
    origin: list[float]
    transition_time: float
    soft_start: bool
    entries: list[dict]

    @classmethod
    def from_config(cls, config: dict) -> TrajectorySetConfig:
        """Read the trajectory set the experiment section of a config names."""
        experiment = config["experiment"]
        name = experiment["trajectories"]
        if "trajectory_sets" in config:
            sets = config["trajectory_sets"]
        else:
            sets = toml.load(TRAJECTORIES_CONFIG)["trajectory_sets"]
        if name not in sets:
            raise KeyError(f"No trajectory set {name}, choose one of {sorted(sets)}")
        return cls(
            trajectories=name,
            origin=experiment["origin"],
            transition_time=experiment["transition_time"],
            soft_start=experiment["soft_start"],
            entries=sets[name],
        )

    def build(self) -> CompositeTrajectory:
        """Build the trajectories and chain them, compiled for the flight."""
        trajectories = tuple(
            build_trajectory(origin=self.origin, **entry) for entry in self.entries
        )
        trajectory = CompositeTrajectory(trajectories, self.transition_time)
        trajectory.warm_up(self.start_time)
        return trajectory
