"""Attitude controller tracking a fixed figure-8 through the collective thrust interface.

Position tracking uses a PID law with gravity compensation, and the reference comes from a cubic
spline through predefined waypoints rather than a planner.
"""

from __future__ import annotations  # Python 3.10 type hints

import pickle
from typing import TYPE_CHECKING

import jax
import numpy as np
from crazyflow.dynamics import Dynamics, load_params

from crazyflow_experiments.control.controller import Controller
from crazyflow_experiments.core.agents.bptt_agent import Agent as BPTTAgent
from crazyflow_experiments.core.agents.ppo_agent import Agent as PPOAgent
from crazyflow_experiments.figure8 import MODEL_DIR, checkpoint_name

AGENTS = {"ppo": PPOAgent, "bptt": BPTTAgent}

# Widths of the observation entries that do not depend on the lookahead length
FIXED_OBS_DIMS = {"ang_vel": 3, "last_action": 4, "pos": 3, "quat": 4, "vel": 3}

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from crazyflow_experiments.control.trajectories import Trajectory


class AttitudeRL(Controller):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(
        self,
        initial_obs: dict[str, NDArray[np.floating]],
        trajectory: Trajectory,
        algorithm: str,
        drone_model: str,
        ctrl_freq: float,
        trajectory_time: float,
        dr_scale: float = 0.0,
    ):
        """Initialize the attitude controller.

        Args:
            initial_obs: Observation the drone starts from.
            trajectory: The trajectory the drone follows.
            algorithm: Algorithm the policy was trained with, selecting the checkpoint.
            drone_model: Drone model.
            ctrl_freq: Control frequency.
            trajectory_time: Loop time of the trajectory in s, selecting the policy trained for it.
            dr_scale: Domain randomization scale of the policy, 0 for the nominal one.
        """
        self.freq = ctrl_freq

        drone_params = load_params(Dynamics.first_principles, drone_model)
        self.drone_mass = drone_params["mass"]  # alternatively from sim.drone_mass
        self.thrust_min = drone_params["thrust_min"] * 4  # min total thrust
        self.thrust_max = drone_params["thrust_max"] * 4  # max total thrust

        # Load the policy trained for this trajectory
        self.algo_name = algorithm + "_f8"
        model_path = MODEL_DIR / checkpoint_name(self.algo_name, trajectory_time, dr_scale)
        if not model_path.exists():
            preset = f"{trajectory_time:g}s".replace(".", "_")
            raise FileNotFoundError(
                f"No {algorithm} policy for the {trajectory_time:g} s figure-8 at {model_path}. "
                f"Train it first: train_{algorithm}.py --preset {preset}"
                + (f" --dr_scale {dr_scale:g}" if dr_scale > 0 else "")
            )
        with open(model_path, "rb") as f:
            params = pickle.load(f)
            hidden_size = params["actor"]["params"]["Dense_0"]["kernel"].shape[1]
            num_layers = len(params["actor"]["params"].keys()) - 1
            if "actor_logstd" in params["actor"]["params"].keys():
                num_layers -= 1
            obs_dim = params["actor"]["params"]["Dense_0"]["kernel"].shape[0]
            act_dim = params["actor"]["params"][f"Dense_{num_layers}"]["kernel"].shape[1]
        assert algorithm in AGENTS, f"Unknown algorithm name: {self.algo_name}"
        agent = AGENTS[algorithm].create(
            key=jax.random.PRNGKey(0), obs_dim=obs_dim, act_dim=act_dim, hidden_size=hidden_size
        )
        self.agent = agent.replace(actor_states=agent.actor_states.replace(params=params["actor"]))
        self.last_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        samples_dim = obs_dim - sum(FIXED_OBS_DIMS.values())
        assert samples_dim > 0 and samples_dim % 3 == 0, f"{model_path.name} expects {obs_dim=}"
        self._N = samples_dim // 3
        self._dt = params["samples_dt"]
        self.trajectory = trajectory
        self._t_start = initial_obs["t"]
        self._t_end = trajectory.duration

        # JIT compile so we don't pay the compilation cost at every step
        self.agent.mean_action(self.agent.actor_states.params, self._obs_rl(initial_obs)[None])

    def __call__(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.

        Returns:
            The collective thrust and orientation [r_des, p_des, y_des, t_des] as a numpy array.
        """
        act = self.agent.mean_action(self.agent.actor_states.params, self._obs_rl(obs)[None])
        act = np.array(act)
        act[..., 2] = 0.0  # Enforce yaw to zero
        self.last_action = act.squeeze(0)
        act = self._scale_actions(act.squeeze(0)).astype(np.float32)
        return act

    def _obs_rl(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Extract the relevant parts of the observation for the RL policy."""
        obs_rl = {k: obs[k] for k in FIXED_OBS_DIMS if k != "last_action"}
        # A rotation has the two representations q and -q. The estimator settles on either one when
        # it starts while the simulation always reports w >= 0, and the policy reads the quaternion
        # raw, so the same attitude reaches it as two different inputs unless the sign is pinned.
        quat = np.asarray(obs_rl["quat"], dtype=np.float32)
        obs_rl["quat"] = -quat if quat[3] < 0 else quat
        waypoints_time = np.arange(0, self._N) * self._dt + obs["t"] - self._t_start
        des_pos = np.asarray(self.trajectory.pos(waypoints_time))
        obs_rl["local_samples"] = (des_pos - obs["pos"]).reshape(-1)  # (n_samples * 3,)
        obs_rl["last_action"] = self.last_action  # (4,)
        # alphabetical key order: important for flax policy
        return np.concatenate([obs_rl[k] for k in sorted(obs_rl)], axis=-1).astype(np.float32)

    def _scale_actions(self, actions: NDArray) -> NDArray:
        """Rescale and clip actions from [-1, 1] to [action_sim_low, action_sim_high]."""
        thrust_diff = (self.thrust_max - self.thrust_min) / 2.0
        scale = np.array([np.pi / 2, np.pi / 2, np.pi / 2, thrust_diff], dtype=np.float32)
        thrust_mean = (self.thrust_max + self.thrust_min) / 2.0
        mean = np.array([0.0, 0.0, 0.0, thrust_mean], dtype=np.float32)
        return np.clip(actions, -1.0, 1.0) * scale + mean

    def done(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return True once the planned trajectory duration has elapsed."""
        return obs["t"] - self._t_start >= self._t_end
