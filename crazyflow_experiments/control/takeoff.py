"""Trajectory from the ground to a hover position."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.control.simulator import extract_obs
from crazyflow_experiments.control.trajectories import Trajectory

if TYPE_CHECKING:
    from crazyflow import Sim
    from jax import Array
    from numpy.typing import NDArray


@dataclass(frozen=True)
class TakeoffTrajectory(Trajectory):
    """Straight line from the start pose to the target pose over `duration`, held afterwards.

    The velocity and acceleration are zero, the ramp is slow enough to track from the position.
    """

    pos_start: Array
    yaw_start: float
    pos_end: Array
    yaw_end: float
    duration: float = 4.0

    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        progress = jnp.clip(jnp.asarray(t) / self.duration, 0.0, 1.0)[..., None]
        return self.pos_start + (self.pos_end - self.pos_start) * progress

    def vel(self, t: Array) -> Array:
        """Return zero velocity, shape (..., 3)."""
        return jnp.zeros(jnp.shape(t) + (3,))

    def acc(self, t: Array) -> Array:
        """Return zero acceleration, shape (..., 3)."""
        return jnp.zeros(jnp.shape(t) + (3,))

    def yaw(self, t: Array) -> Array:
        """Return the heading at time t."""
        progress = jnp.clip(jnp.asarray(t) / self.duration, 0.0, 1.0)
        return self.yaw_start + (self.yaw_end - self.yaw_start) * progress

    def setpoints(
        self, ctrl_freq: float, hover_time: float = 2.0
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Sample the ramp and the hover of `hover_time` after it at the control rate.

        Args:
            ctrl_freq: Rate at which the setpoints are sent.
            hover_time: Seconds the target pose is held after the ramp.

        Returns:
            The positions as (N, 3) and the yaws as (N,).
        """
        t = np.arange(int(ctrl_freq * (self.duration + hover_time)) + 1) / ctrl_freq
        return np.asarray(self.pos(t)), np.asarray(self.yaw(t))


def takeoff(
    sim: Sim,
    target_pos: NDArray[np.floating],
    target_yaw: float,
    ctrl_freq: float = 50.0,
    duration: float = 4.0,
    hover_time: float = 2.0,
    render: bool = False,
):
    """Fly the drone to its target position.

    Args:
        sim: Crazyflow simulation.
        target_pos: Target position as (3,).
        target_yaw: Target yaw in radians.
        ctrl_freq: Control frequency.
        duration: Duration of the takeoff.
        hover_time: Hover duration.
        render: Whether to render the simulation.
    """
    pos_start = np.array(sim.data.states.pos[0, 0])
    yaw_start = R.from_quat(np.array(sim.data.states.quat[0, 0])).as_euler("xyz")[2]
    trajectory = TakeoffTrajectory(
        pos_start, yaw_start, np.asarray(target_pos), target_yaw, duration
    )
    trajectory_pos, trajectory_yaw = trajectory.setpoints(ctrl_freq, hover_time)
    if sim.control != Control.state:
        controller = MellingerController(extract_obs(sim), trajectory, sim.drone)

    cmd = np.zeros(16)
    for pos, yaw in zip(trajectory_pos, trajectory_yaw, strict=True):
        if sim.control == Control.state:
            cmd[:3], cmd[9:13] = pos, R.from_euler("z", yaw).as_quat()
            sim.state_control(np.broadcast_to(cmd, (sim.n_worlds, sim.n_drones, 16)))
        else:
            action = controller(extract_obs(sim))
            sim.attitude_control(np.broadcast_to(action, (sim.n_worlds, sim.n_drones, 4)))
        if render:
            sim.render()
        sim.step(sim.freq // ctrl_freq)
