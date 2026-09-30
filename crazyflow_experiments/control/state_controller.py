"""Controller that plays a trajectory through the state interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from crazyflow_experiments.control.trajectories import Trajectory


class StateController(Controller):
    """Sends the reference state of the trajectory and leaves the tracking to the drone."""

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], trajectory: Trajectory):
        """Initialize the controller.

        Args:
            initial_obs: The initial observation, its time `t` starts the trajectory.
            trajectory: The trajectory the drone follows.
        """
        self.trajectory = trajectory
        self._t_start = initial_obs["t"]
        self._t_end = trajectory.duration

    def __call__(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Return the state command [pos, vel, acc, quat, body_rates] at obs["t"]."""
        t = obs["t"] - self._t_start
        yaw = float(self.trajectory.yaw(t))  # To not trigger a JAX compile in R
        return np.concatenate(
            (
                self.trajectory.pos(t),
                self.trajectory.vel(t),
                self.trajectory.acc(t),
                R.from_euler("z", yaw).as_quat(),
                np.zeros(3),
            ),
            dtype=np.float64,
        )

    def done(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return True once the planned trajectory duration has elapsed."""
        return obs["t"] - self._t_start >= self._t_end
