"""Multi-modal MPPI/CEM controller.

K controllers run in parallel, each with its own mean and variance. Every controller averages its
own elite samples, and the best controller supplies the executed action.
"""

from __future__ import annotations  # Python 3.10 type hints

import os

os.environ["SCIPY_ARRAY_API"] = "1"

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.dynamics import load_params
from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.functional import attitude_control
from einops import rearrange
from jax import vmap
from jax.random import truncated_normal

from crazyflow_experiments.control.config import SOFT_START_TIME
from crazyflow_experiments.control.controller import Controller

if TYPE_CHECKING:
    from collections.abc import Callable

    from crazyflow.sim.data import SimData
    from jax import Array
    from numpy.typing import NDArray

    from crazyflow_experiments.control.trajectories import Trajectory

DIVERGED_COST = 1e6


class AttitudeMPPIController(Controller):
    """Multi-modal MPPI/CEM controller."""

    def __init__(
        self,
        initial_obs: dict[str, NDArray[np.floating]],
        initial_info: dict,
        trajectory: Trajectory,
        soft_start: bool = False,
    ):
        """Build the rollout simulator and the sampling distribution of every mode."""
        self.N = initial_info["controller"]["mppi"]["N"]
        self.T = initial_info["controller"]["mppi"]["T"]
        self.dt = self.T / self.N
        # The cost computes after a step, so the horizon goes from 1 to N + 1
        self.horizon_times = jnp.arange(1, self.N + 1) * self.dt
        assert self.N % self.T == 0, "N must be divisible by T"
        self.f = int(self.N / self.T)
        self.ctrl_dt = 1 / initial_info["controller"]["ctrl_freq"]
        self.n_samples = initial_info["controller"]["mppi"]["n_samples"]
        self.K = initial_info["controller"]["mppi"]["K"]
        self.M = self.n_samples // self.K  # samples per mode

        self.sim = Sim(
            n_worlds=self.n_samples,
            n_drones=1,
            attitude_freq=self.f,
            freq=self.f,
            dynamics=Dynamics.so_rpy_rotor_drag,
            control=Control.attitude,
            drone=initial_info["drone_model"],
            device="gpu",
        )
        self.sim.reset()

        sigma = initial_info["controller"]["mppi"]["noise_sigma"]
        self.noise_sigmas = jnp.full((self.K, self.N, 4), fill_value=sigma, device=self.sim.device)
        self.temperature = initial_info["controller"]["mppi"]["temperature"]
        self.elite_percentage = initial_info["controller"]["mppi"]["elite_percentage"]
        self.beta = initial_info["controller"]["mppi"]["beta"]
        self.alpha = initial_info["controller"]["mppi"]["alpha"]
        self.min_variance = initial_info["controller"]["mppi"]["min_variance"]
        self.mean_controls = jnp.zeros((self.K, self.N, 4), device=self.sim.device)
        self.key = jax.random.PRNGKey(0)
        # Shape: (n_obstacles, 3)
        self.obstacles = jnp.array(initial_info["obstacles"], device=self.sim.device)
        self.drone_params = load_params(Dynamics.so_rpy_rotor_drag, initial_info["drone_model"])
        self.thrust_time_coef = float(self.sim.data.params.thrust_time_coef[0])
        self.act_low = -jnp.ones(4, device=self.sim.device) * jnp.pi / 2
        self.act_low = self.act_low.at[3].set(self.drone_params["thrust_min"] * 4)
        self.act_high = jnp.ones(4, device=self.sim.device) * jnp.pi / 2
        self.act_high = self.act_high.at[3].set(self.drone_params["thrust_max"] * 4)
        self.thrust = np.zeros(4)

        self.soft_start = soft_start
        self.trajectory = trajectory
        self._t_start = initial_obs["t"]
        self._t_end = trajectory.duration

        gravity = -self.drone_params["gravity_vec"][2]
        self.control_fn = build_compute_control_fn(
            self.sim,
            obstacles=self.obstacles,
            collision_radius=initial_info["obstacle_radius"] + initial_info["drone_radius"],
            weights=initial_info["controller"]["cost"],
            hover_thrust=self.drone_params["mass"] * gravity,
            act_low=self.act_low,
            act_high=self.act_high,
            n_modes=self.K,
            n_samples=self.M,
            beta=self.beta,
            elite_percentage=self.elite_percentage,
            temperature=self.temperature,
            alpha=self.alpha,
            min_variance=self.min_variance,
            dt_plan=self.dt,
            dt_ctrl=self.ctrl_dt,
        )

        self(initial_obs)  # Warm up the controller

    def __call__(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.

        Returns:
            The collective thrust and orientation [r_des, p_des, y_des, t_des] as a numpy array.
        """
        obs["rotor_vel"] = self.thrust
        obs_device = {k: jax.device_put(v, self.sim.device) for k, v in obs.items()}
        t = obs["t"] - self._t_start
        horizon = t + self.horizon_times
        refs = {
            "pos": jax.device_put(self.trajectory.pos(horizon), self.sim.device),
            "vel": jax.device_put(self.trajectory.vel(horizon), self.sim.device),
            "acc": jax.device_put(self.trajectory.acc(horizon), self.sim.device),
            "yaw": jax.device_put(self.trajectory.yaw(horizon), self.sim.device),
        }

        self.key, key = jax.random.split(self.key)
        action, means, sigmas, best_mode_idx, costs, positions = self.control_fn(
            key, obs_device, refs, self.mean_controls, self.noise_sigmas
        )
        self.mean_controls, self.noise_sigmas = means, sigmas
        # Store for visualization outside of JIT
        self.best_mode_idx, self.costs, self.positions = best_mode_idx, costs, positions

        action = np.asarray(action)  # Convert to numpy for deployment
        self.thrust += (action[3] - self.thrust) / self.thrust_time_coef * self.ctrl_dt
        return action

    def start(self, t: float):
        """Set the time the tracked trajectory begins at."""
        self._t_start = t + (SOFT_START_TIME if self.soft_start else 0.0)

    def elapsed(self, t: float) -> float:
        """Return the time since start."""
        return t - self._t_start

    def reference_positions(self, t: float) -> NDArray[np.floating]:
        """Return the planner reference positions."""
        horizon = t - self._t_start + self.horizon_times
        return np.asarray(self.trajectory.pos(horizon), dtype=np.float32)

    def done(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return True if the controller is done."""
        return obs["t"] - self._t_start >= self._t_end


def build_compute_control_fn(
    sim: Sim,
    *,
    obstacles: Array,
    collision_radius: float,
    weights: dict[str, float],
    hover_thrust: float,
    act_low: Array,
    act_high: Array,
    n_modes: int,
    n_samples: int,
    beta: float,
    elite_percentage: float,
    temperature: float,
    alpha: float,
    min_variance: float,
    dt_plan: float,
    dt_ctrl: float,
) -> Callable:
    """Build the MPPI control function with the sim and all settings fused into it."""
    step_fn = sim.build_step_fn()
    n_steps = sim.freq // sim.control_freq
    sim_data = sim.data
    update = vmap(partial(update_mode, elite_percentage, temperature, alpha, min_variance))
    cost_fn = partial(compute_cost, obstacles, collision_radius, weights, hover_thrust)

    def mppi_step(
        data: SimData, info: tuple[Array, dict[str, Array]]
    ) -> tuple[SimData, tuple[Array, Array]]:
        """Apply one control command to all rollouts and step the sim forward."""
        cmd, ref = info
        data = step_fn(attitude_control(data, cmd[:, None, :]), n_steps)
        pos = data.states.pos
        # If pathological rollouts diverge, we set their cost to avoid NaNs corrupting the mode
        finite = jnp.all(jnp.isfinite(pos), axis=(-2, -1))
        cost = cost_fn(data, ref)
        cost = jnp.where(finite & jnp.isfinite(cost), cost, DIVERGED_COST)
        return data, (cost, jnp.where(finite[:, None, None], pos, ref["pos"]))

    @jax.jit
    def compute_control(
        key: Array, obs: dict[str, Array], refs: dict[str, Array], means: Array, sigmas: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Sample control candidates, roll them out, and update the sampling distributions.

        Returns:
            The action to execute, the means and standard deviations shifted to the next control
            cycle, the index of the mode with the lowest cost, and the sample costs and positions of
            all modes for visualization.
        """
        # Zero-mean noise for every mode, truncated so that mean + noise are within action limits
        lower, upper = (act_low - means) / sigmas, (act_high - means) / sigmas
        shape = (n_modes, n_samples, *means.shape[1:])
        noise = truncated_normal(key, lower[:, None], upper[:, None], shape) * sigmas[:, None]

        # y[t] = beta * y[t-1] + (1 - beta) * x[t] with y[-1] = 0 is a causal linear filter, i.e. a
        # matmul along the horizon with the lower triangular kernel (1 - beta) * beta**(t - i)
        lag = jnp.arange(means.shape[1])[:, None] - jnp.arange(means.shape[1])[None, :]
        smoothing = jnp.where(lag >= 0, (1 - beta) * beta ** jnp.abs(lag), 0)
        noise = smoothing @ noise

        # The dynamics do not care about modes, so we flatten them into the rollout batch
        controls = rearrange(means[:, None] + noise, "k m n u -> n (k m) u")
        _, (costs, positions) = jax.lax.scan(mppi_step, set_state(sim_data, obs), (controls, refs))
        costs = rearrange(jnp.sum(costs, axis=0), "(k m) -> k m", k=n_modes)
        positions = rearrange(positions, "n (k m) 1 p -> k m n p", k=n_modes)

        means, sigmas, mode_costs = update(means, noise, costs, sigmas)
        best_mode = jnp.argmin(mode_costs)
        # We only execute the first action of the mode with the lowest cost, and shift the horizon
        # of all modes forward by one control cycle for the next update
        return (
            means[best_mode, 0],
            shift_controls(means, dt_plan, dt_ctrl),
            shift_controls(sigmas, dt_plan, dt_ctrl),
            best_mode,
            costs,
            positions,
        )

    return compute_control


def update_mode(
    elite_percentage: float,
    temperature: float,
    alpha: float,
    min_variance: float,
    mean: Array,
    noise: Array,
    costs: Array,
    sigma: Array,
) -> tuple[Array, Array, Array]:
    """Update the sampling distribution of a single mode from its elite samples."""
    n_elites = int(costs.shape[0] * elite_percentage)
    assert n_elites > 0, "elite_percentage is too small for the number of samples per mode"
    elite_idx = jnp.argsort(costs)[:n_elites]
    elite_noise, elite_costs = noise[elite_idx], costs[elite_idx]
    min_cost = jnp.min(elite_costs)
    weights = jnp.exp(-(elite_costs - min_cost) / temperature)
    weights = weights[:, None, None] / jnp.sum(weights)
    delta = jnp.sum(weights * elite_noise, axis=0)
    variance = jnp.sum(weights * (elite_noise - delta) ** 2, axis=0)
    sigma = jnp.sqrt(jnp.maximum(alpha * variance + (1 - alpha) * sigma**2, min_variance))
    return mean + delta, sigma, min_cost


def shift_controls(controls: Array, dt_plan: float, dt_ctrl: float) -> Array:
    """Shift a control trajectory forward in time by one control cycle.

    Since the time for planning and for control are different, we interpolate the new controls.
    """
    horizon, shift = controls.shape[-2], dt_ctrl / dt_plan
    lower = jnp.minimum(jnp.arange(horizon) + int(shift), horizon - 1)
    upper = jnp.minimum(lower + 1, horizon - 1)
    frac = shift - int(shift)
    return (1 - frac) * controls[..., lower, :] + frac * controls[..., upper, :]


def set_state(data: SimData, obs: dict[str, Array]) -> SimData:
    """Set the state of all rollouts to the current state of the drone."""
    return data.replace(
        states=data.states.replace(
            pos=data.states.pos.at[...].set(obs["pos"]),
            quat=data.states.quat.at[...].set(obs["quat"]),
            vel=data.states.vel.at[...].set(obs["vel"]),
            ang_vel=data.states.ang_vel.at[...].set(obs["ang_vel"]),
            rotor_vel=data.states.rotor_vel.at[...].set(obs["rotor_vel"]),
        )
    )


def compute_cost(
    obstacles: Array,
    collision_radius: float,
    weights: dict[str, float],
    hover_thrust: float,
    data: SimData,
    reference: dict[str, Array],
) -> Array:
    """Compute the cost of a single rollout step."""
    pos, vel = data.states.pos[:, 0, :], data.states.vel[:, 0, :]
    cmd = data.controls.attitude.staged_cmd[:, 0, :]
    des_pos, des_vel = reference["pos"][None, :], reference["vel"][None, :]
    des_yaw = reference["yaw"][None]

    # Tracking cost, with an additional penalty on the altitude error
    pos_cost = jnp.sum((pos - des_pos) ** 2, axis=-1) * weights["position"]
    z_cost = jnp.abs(pos[:, 2] - des_pos[:, 2]) * weights["altitude"]
    vel_cost = jnp.sum((vel - des_vel) ** 2, axis=-1) * weights["velocity"]
    state_cost = pos_cost + z_cost + vel_cost

    # Input cost, penalizing tilt, thrust deviations from gravity and yaw deviations
    tilt_cost = jnp.sum(cmd[:, :2] ** 2, axis=-1) * weights["tilt"]
    thrust_cost = (cmd[:, 3] - hover_thrust) ** 2 * weights["thrust"]
    yaw_cost = (cmd[:, 2] - des_yaw) ** 2 * weights["yaw"]
    input_cost = tilt_cost + thrust_cost + yaw_cost

    # Obstacles are modeled as vertical cylinders, so we only check the distance in the xy plane
    distance = jnp.linalg.norm(pos[:, None, :2] - obstacles[None, :, :2], axis=-1)
    obstacle_cost = weights["obstacle"] * jnp.sum(distance < collision_radius, axis=-1)

    return state_cost + input_cost + obstacle_cost
