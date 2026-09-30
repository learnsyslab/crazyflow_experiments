"""Interface shared by every controller that flies a drone."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class Controller(ABC):
    """Base class for controller implementations."""

    @abstractmethod
    def __call__(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Compute the action to send to the drone.

        Args:
            obs: Current observation of the drone state.

        Returns:
            A state command [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz] for
            the state interface, or an attitude command [roll, pitch, yaw, thrust] for the attitude
            interface.
        """

    def done(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Report whether the controller has finished.

        Args:
            obs: Current observation of the drone state.

        Returns:
            True once the controller is done.
        """
        return False
