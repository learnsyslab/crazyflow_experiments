"""TD3 controller for the learn2fly reach-position task.

The observation layout and action scaling must match training, so keys are concatenated
alphabetically and actions are mapped back to RPM by inverting NormalizeActionsQuadratic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.controller import Controller
from crazyflow_experiments.core.agents.td3_agent import load_actor, load_actor_params
from crazyflow_experiments.core.envs.drone_env import HOVER_RPM, create_action_space
from crazyflow_experiments.core.envs.wrappers import quadratic_action_coefficients
from crazyflow_experiments.learn2fly import MODEL_DIR

# Widths of the observation entries that do not depend on the action history length
FIXED_OBS_DIMS = {"ang_vel": 3, "pos": 3, "rot_mat": 9, "vel": 3}

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from crazyflow_experiments.control.trajectories import Trajectory


class Learn2FlyRL(Controller):
    """TD3 controller for the learn2fly reach-position task."""

    def __init__(
        self,
        initial_obs: dict[str, NDArray[np.floating]],
        trajectory: Trajectory,
        max_duration: float = 15.0,
    ):
        """Initialize the controller from the reset observation and environment information.

        Args:
            initial_obs: Observation the drone starts from.
            trajectory: The trajectory whose current point the policy flies to.
            max_duration: Time after which the controller reports that it has finished.
        """
        drone: str = "cf2x_L250"
        self.trajectory = trajectory
        self.goal_pos = np.zeros(3, dtype=np.float32)
        self._k0, self._k1, self._k2 = self._compute_scaling_coefficients(drone, HOVER_RPM)

        model_path = MODEL_DIR / "td3_rprv_model.ckpt"
        self.agent = load_actor(model_path)
        act_dim = self.agent.actor.act_dim
        obs_dim = load_actor_params(model_path)["params"]["Dense_0"]["kernel"].shape[0]

        actions_dim = obs_dim - sum(FIXED_OBS_DIMS.values())
        assert actions_dim > 0 and actions_dim % act_dim == 0, f"expected {obs_dim=}"
        self.num_last_actions = actions_dim // act_dim

        # Ring buffer of past actions, newest at index 0
        self.last_actions = np.zeros((self.num_last_actions, act_dim), dtype=np.float32)

        # Warm up JAX compilation
        self.agent.mean_action(self.agent.actor_states.params, self._obs(initial_obs)[None])

        self._t_start: float = 0.0
        self._max_duration = max_duration

    @staticmethod
    def _compute_scaling_coefficients(
        drone: str, neutral_rpm: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the coefficients that invert the training action mapping.

        The bounds come from the drone parameters, since no environment exists on the real drone.

        Args:
            drone: Drone model to load the RPM bounds from.
            neutral_rpm: Hover RPM setpoint mapped to action 0.

        Returns:
            The coefficients k0, k1 and k2, each of shape (4,).
        """
        action_space = create_action_space("rotor_vel", drone)
        coefficients = quadratic_action_coefficients(
            action_space.low, action_space.high, np.full(4, neutral_rpm)
        )
        # Back to numpy, so the per-step rescaling stays off the JAX dispatch path
        return tuple(np.asarray(k, dtype=np.float32) for k in coefficients)

    def __call__(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Compute rotor velocity setpoints from the current observation.

        Args:
            obs: Observation with keys pos, quat, vel, ang_vel and t.

        Returns:
            Rotor velocity setpoints of shape (4,) in [rpm_min, rpm_max].
        """
        self.goal_pos = np.asarray(self.trajectory.pos(obs["t"] - self._t_start))
        obs_vec = self._obs(obs)
        act = self.agent.mean_action(self.agent.actor_states.params, obs_vec[None])
        act = np.array(act).squeeze(0)
        self.last_actions = np.roll(self.last_actions, shift=1, axis=0)
        self.last_actions[0] = act
        return self._scale_actions(act)

    def _obs(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Build the flat actor observation of shape (actor_obs_dim,) from a drone observation.

        The keys are concatenated in alphabetical order to match the order in FlattenJaxObservation.
        """
        obs = {
            "ang_vel": obs["ang_vel"],
            "last_actions": self.last_actions.reshape(-1),
            "pos": obs["pos"] - self.goal_pos,
            "rot_mat": R.from_quat(obs["quat"]).as_matrix().reshape(-1),
            "vel": obs["vel"],
        }
        return np.concatenate([obs[key] for key in sorted(obs)]).astype(np.float32)

    def _scale_actions(self, actions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Map normalized actions in [-1, 1] to rotor velocity setpoints of shape (4,)."""
        a = np.clip(actions, -1.0, 1.0).astype(np.float32)
        return (self._k0 + self._k1 * a + self._k2 * a**2).astype(np.float32)

    def done(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return True when the drone leaves the safe volume or max duration is reached."""
        return not self._safety_check(obs) or obs["t"] - self._t_start >= self._max_duration

    def _safety_check(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        pos = obs["pos"]
        return bool(np.all(np.abs(pos) <= 3.0) and pos[2] <= 2.5)
