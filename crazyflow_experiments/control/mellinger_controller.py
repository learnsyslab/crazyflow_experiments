"""Mellinger-style trajectory tracking through the collective thrust and attitude interface.

A PID law on the position error with gravity compensation gives the desired thrust vector, whose
direction and the reference yaw define the desired attitude.
"""

from __future__ import annotations  # Python 3.10 type hints

import os
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.control.config import drone_params
from crazyflow_experiments.control.controller import Controller

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from crazyflow_experiments.control.trajectories import Trajectory


class MellingerController(Controller):
    """Mellinger position controller commanding collective thrust and attitude.

    Modified from https://github.com/utiasDSL/crazyswarm-import/blob/ad2f7ea987f458a504248a1754b124ba39fc2f21/ros_ws/src/crazyswarm/scripts/position_ctl_m.py
    """

    def __init__(
        self,
        initial_obs: dict[str, NDArray[np.floating]],
        trajectory: Trajectory,
        drone_model: str,
        low_level_ctrl_freq: float = 500,
        tilt_max: float | None = None,
        legacy_heading: bool = False,
    ):
        """Initialization of the controller.

        Args:
            initial_obs: The initial observation, its time `t` starts the trajectory.
            trajectory: The trajectory the drone follows.
            drone_model: Drone model, whose mass comes from `sysid/config_drones.toml`.
            low_level_ctrl_freq: Rate of the onboard attitude controller.
            tilt_max: Limit of the commanded roll and pitch in rad, None for no limit.
            legacy_heading: Build the attitude like the firmware, from the x axis of the desired
                heading, so that the yaw turns with the tilt. The system identification data was
                collected this way. Otherwise the y axis of the heading is used, which keeps the yaw.
        """
        self.low_level_ctrl_freq = low_level_ctrl_freq
        self.tilt_max = tilt_max
        self.legacy_heading = legacy_heading
        self.drone_mass = drone_params(drone_model)["mass"]
        match drone_model:
            case _ if "cf" in drone_model:
                self.kp = np.array([0.4, 0.4, 1.25])
                self.ki = np.array([0.05, 0.05, 0.05])
                self.kd = np.array([0.2, 0.2, 0.4])
                self.ki_range = np.array([2.0, 2.0, 0.4])
            case "racer":
                self.kp = np.array([3.8, 3.8, 3.4])
                self.ki = np.array([0.00, 0.00, 0.01])
                self.kd = np.array([1.5, 1.5, 1.4])
                self.ki_range = np.array([2.0, 2.0, 0.4])
            case "hb_x500":
                self.kp = np.array([12.0, 12.0, 35.0])
                self.ki = np.array([1.00, 1.00, 1.00])
                self.kd = np.array([6.0, 6.0, 8.0])
                self.ki_range = np.array([4.0, 4.0, 2.0])
            case _:
                raise ValueError(f"Unknown drone model: {drone_model}")

        self.i_error = np.zeros(3)
        self.g = 9.81

        self.trajectory = trajectory
        self._t_start = initial_obs["t"]
        self._t_end = trajectory.duration

    def __call__(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.

        Returns:
            The orientation and collective thrust [roll, pitch, yaw, thrust] as a numpy array.
        """
        t = obs["t"] - self._t_start

        des_pos = np.asarray(self.trajectory.pos(t))
        des_vel = np.asarray(self.trajectory.vel(t))
        des_acc = np.asarray(self.trajectory.acc(t))
        des_yaw = float(self.trajectory.yaw(t))

        # Calculate the deviations from the desired trajectory
        pos, vel = obs["pos"], obs["vel"]
        pos_error = des_pos - pos
        vel_error = des_vel - vel

        # Update integral error
        self.i_error += pos_error * (1 / self.low_level_ctrl_freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust += self.drone_mass * des_acc
        target_thrust[2] += self.drone_mass * self.g

        # Update z_axis to the current orientation of the drone
        z_axis = R.from_euler("xyz", obs["rpy"]).as_matrix()[:, 2]

        # update current thrust
        currentThrust = target_thrust.dot(z_axis)
        currentThrust = max(currentThrust, 0.3 * self.drone_mass * self.g)
        currentThrust = min(currentThrust, 1.8 * self.drone_mass * self.g)

        # Desired attitude from the thrust direction and the heading
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        if self.legacy_heading:
            x_c_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
            y_axis_desired = np.cross(z_axis_desired, x_c_des)
            y_axis_desired /= np.linalg.norm(y_axis_desired)
            x_axis_desired = np.cross(y_axis_desired, z_axis_desired)
        else:
            y_c_des = np.array([-np.sin(des_yaw), np.cos(des_yaw), 0.0])
            x_axis_desired = np.cross(y_c_des, z_axis_desired)
            x_axis_desired /= np.linalg.norm(x_axis_desired)
            y_axis_desired = np.cross(z_axis_desired, x_axis_desired)
            y_axis_desired /= np.linalg.norm(y_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)
        if self.tilt_max is not None:
            euler_desired[:2] = np.clip(euler_desired[:2], -self.tilt_max, self.tilt_max)
        return np.concatenate([euler_desired, [currentThrust]])

    def done(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return True once the planned trajectory duration has elapsed."""
        return obs["t"] - self._t_start >= self._t_end
