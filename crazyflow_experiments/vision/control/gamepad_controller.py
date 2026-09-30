"""Manual attitude controller using a gamepad.

To run it, you need to use the gamepad environment. You can connect any gamepad (here for an XBOX
controller) and fly. Note that there is cover support to simplify the controlls. You might also want
to set the camera to fpv in the toml file.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import pygame
from crazyflow.dynamics import Dynamics
from crazyflow.dynamics import load_params as load_drone_params

from crazyflow_experiments.control.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GamepadController(Controller):
    """Manual attitude controller using a gamepad."""

    def __init__(self, drone_model: str, ctrl_freq: float):
        """Initialize the gamepad controller.

        Args:
            drone_model: Drone model.
            ctrl_freq: Control frequency.
        """
        self.freq = ctrl_freq

        # For more info on the models, check out https://github.com/learnsyslab/drone-models
        drone_params = load_drone_params(Dynamics.first_principles, drone_model)
        self.drone_mass = drone_params["mass"] + 20.1 * 1e-3  # fpv drone weighs 63.5g, base is 43.4
        self.thrust_max = drone_params["thrust_max"] * 4  # Total thrust, trimmed by the triggers
        self.g = 9.81
        self._roll = 0
        self._pitch = 0
        self._yaw = 0

        pygame.init()
        pygame.joystick.init()
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        print("Controller:", self._joystick.get_name())

        self._finished = False

    def __call__(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        thrust_desired = self.drone_mass * self.g  # Compute target thrust
        # update z_axis_desired
        pygame.event.pump()

        for i in range(self._joystick.get_numbuttons()):
            if self._joystick.get_button(i):
                print("Button", i, "pressed")
                if i == 0:  # A button = emergency stop
                    self._finished = True

        deadzone = 0.1
        stick_l_x = self._apply_deadzone(self._joystick.get_axis(0), deadzone)
        # stick_l_y = self._apply_deadzone(self._joystick.get_axis(1), deadzone)
        lt = self._apply_deadzone(self._joystick.get_axis(2), deadzone)
        stick_r_x = self._apply_deadzone(self._joystick.get_axis(3), deadzone)
        stick_r_y = self._apply_deadzone(self._joystick.get_axis(4), deadzone)
        rt = self._apply_deadzone(self._joystick.get_axis(5), deadzone)

        max_angle = np.pi / 4
        max_angle_rate = np.pi / self.freq

        self._roll = 0.9 * (self._roll + stick_r_x * max_angle_rate)
        self._pitch = 0.9 * (self._pitch + -stick_r_y * max_angle_rate)
        self._yaw = self._yaw + -stick_l_x * max_angle_rate
        euler = [self._roll, self._pitch, self._yaw]
        euler[:2] = np.clip(euler[:2], -max_angle, max_angle)

        thrust_desired += self.thrust_max / 6 * (rt + 1) / 2
        thrust_desired -= self.thrust_max / 6 * (lt + 1) / 2

        return np.concat([euler, [thrust_desired]], dtype=np.float32)

    def done(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return True if the controller is done."""
        return self._finished

    def _apply_deadzone(self, value: float, deadzone: float = 0.1) -> float:
        if abs(value) < deadzone:
            return 0.0
        return value
