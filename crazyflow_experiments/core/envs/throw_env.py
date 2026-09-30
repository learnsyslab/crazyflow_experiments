"""Crazyflow environment for throwing drones into the air and learning to stabilize."""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"

import functools
from typing import TYPE_CHECKING, Callable

import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
from crazyflow.control import Control
from crazyflow.dynamics import load_params
from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.pipeline import append_fn, remove_fn
from crazyflow.sim.visualize import draw_line
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from jax import Device

# Randomization constants (fully random throw mode)
SPAWN_Z_MIN = 1.8  # m
SPAWN_Z_MAX = 2.5  # m
VEL_MAX = 1.5  # m/s
VEL_ANGLE_MAX = np.pi / 2  # rad, maximum angular disturbance from downward direction
ANG_VEL_MAX = 1.5 * np.pi  # rad/s

# Success termination constants
SUCCESS_Z = 1.8  # m, target stabilization height
SUCCESS_Z_TOL = 0.2  # m
SUCCESS_VEL_TOL = 1.0  # m/s, velocity magnitude threshold
SUCCESS_ANG_VEL_TOL = 0.2  # rad/s, angular velocity magnitude threshold


def create_action_space(drone: str, control_type: Control | str) -> spaces.Box:
    """Select the appropriate action space for a given control type.

    Args:
        control_type: The desired control mode.
        drone: Drone model of the environment.

    Returns:
        The action space.
    """
    params = load_params(Dynamics.first_principles, drone)
    thrust_min, thrust_max = float(params["thrust_min"]) * 4, float(params["thrust_max"]) * 4
    match control_type:
        case Control.attitude:
            return spaces.Box(
                np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, thrust_min], dtype=np.float32),
                np.array([np.pi / 2, np.pi / 2, np.pi / 2, thrust_max], dtype=np.float32),
            )
        case "rotor_thrust":  # Thrust per rotor
            return spaces.Box(float(params["thrust_min"]), float(params["thrust_max"]), shape=(4,))
        case _:
            raise ValueError(f"Unsupported control type: {control_type}")


def thrust2rotor_vel(thrust: Array, rpm2thrust: Array) -> Array:
    """Invert the thrust curve thrust = c0 + c1 * rpm + c2 * rpm^2 for the rotor velocity."""
    c0, c1, c2 = rpm2thrust
    return (-c1 + jp.sqrt(c1**2 - 4 * c2 * (c0 - thrust))) / (2 * c2)


def _obs(data: SimData) -> dict[str, Array]:
    """Extract observations from simulation data."""
    obs = {
        "z": data.states.pos[:, 0, 2],
        "quat": data.states.quat[:, 0, :],
        "vel": data.states.vel[:, 0, :],
        "ang_vel": data.states.ang_vel[:, 0, :],
    }
    return obs


def _reward(
    pos: Array,
    vel: Array,
    height_weight: float,
    vel_weight: float,
    vel_coef: float,
    floor_weight: float,
    floor_coef: float,
    ceiling_weight: float,
    ceiling_coef: float,
    fall_weight_below: float,
    fall_weight_above: float,
) -> Array:
    """Calculate reward based on distance to target height."""
    z = pos[:, 2]
    # Linear in the downward speed, whose gradient the exponential velocity term loses when falling
    # fast. Weighted separately below and above the target height
    fall_speed = jp.maximum(-vel[:, 2], 0.0)
    below_setpoint = z < SUCCESS_Z
    # Linear, so the pull towards the target height does not fade with the distance
    height_reward = -height_weight * jp.abs(z - SUCCESS_Z)
    floor_penalty = floor_weight * -jp.exp(-floor_coef * z)
    ceiling_penalty = ceiling_weight * -jp.exp(ceiling_coef * (z - 4))
    # Penalize downward velocity when below setpoint, upward velocity when above
    above_setpoint = z > SUCCESS_Z
    vel_z_filtered = jp.where(
        above_setpoint, jp.maximum(vel[:, 2], 0.0), jp.minimum(vel[:, 2], 0.0)
    )
    vel = vel.at[:, 2].set(vel_z_filtered)
    norm_vel = jp.linalg.norm(vel, axis=-1)
    vel_reward = vel_weight * jp.exp(-vel_coef * norm_vel)
    reward = height_reward + vel_reward + floor_penalty + ceiling_penalty
    reward = reward - fall_weight_below * jp.where(below_setpoint, fall_speed, 0.0)
    return reward - fall_weight_above * jp.where(below_setpoint, 0.0, fall_speed)


def _terminated(pos: Array, success: Array) -> Array:
    """Check if any environment has terminated."""
    z_min, z_max = 0.05, 4.0
    return (pos[:, 0, 2] < z_min) | (pos[:, 0, 2] > z_max) | jp.isnan(pos[:, 0, 2]) | success


def _truncated(time: Array, max_episode_time: float) -> Array:
    """Check if any environment has reached max episode time."""
    return time >= max_episode_time


def _success(vel: Array, ang_vel: Array, action: Array) -> Array:
    """Check if the drone has stabilized at the target height."""
    vel_ok = jp.linalg.norm(vel, axis=-1) < SUCCESS_VEL_TOL
    ang_vel_ok = jp.linalg.norm(ang_vel, axis=-1) < SUCCESS_ANG_VEL_TOL
    not_nan = jp.logical_not(jp.isnan(action).any(axis=-1))
    return vel_ok & ang_vel_ok & not_nan


def _apply_action(
    data: SimData,
    action: Array,
    action_space: spaces.Box,
    num_envs: int,
    control_type: Control | str,
    rpm2thrust: Array,
) -> SimData:
    """Apply action to simulation data."""
    low, high = action_space.low, action_space.high
    action = jp.clip(action, low, high).reshape((num_envs, 1, -1))
    match control_type:
        case Control.attitude:
            data = data.replace(
                controls=data.controls.replace(
                    attitude=data.controls.attitude.replace(staged_cmd=action)
                )
            )
        case "rotor_thrust":
            rotor_vel = thrust2rotor_vel(action, rpm2thrust)
            data = data.replace(controls=data.controls.replace(rotor_vel=rotor_vel))
        case _:
            raise ValueError(f"Unsupported control type: {control_type}")
    return data


def reset(
    env: ThrowEnv, sim: Sim, device: Device, *, seed: int | None = None, options: dict | None = None
) -> tuple[tuple[SimData, Array, Array], tuple[dict[str, Array], dict]]:
    """Reset the environment."""
    data = env.data
    if seed is not None:
        rng_key = jax.device_put(jax.random.key(seed), device)
        data = data.replace(core=data.core.replace(rng_key=rng_key))
    default_data = _make_default_data(env, sim.default_data)
    data = sim._reset(data, default_data, None)
    _marked_for_reset = env._marked_for_reset.at[...].set(False)
    return env.replace(data=data, _marked_for_reset=_marked_for_reset), (_obs(data), {})


def step(
    env: ThrowEnv, action: Array, sim: Sim, n_substeps: int, rpm2thrust: Array
) -> tuple[tuple[SimData, Array], tuple[Array, Array, Array, Array, dict]]:
    """Step the environment."""
    data, marked_for_reset = env.data, env._marked_for_reset
    # 1. apply action: only attitude control
    data = _apply_action(data, action, env.action_space, env.num_envs, env.control_type, rpm2thrust)
    # 2. step sim for n_substeps
    data = sim._step(data, n_substeps)
    # 3. handle autoreset & update mask
    default_data = _make_default_data(env, sim.default_data)
    # The reset randomization runs for all worlds, so skip it in the steps where none restarts
    data = jax.lax.cond(
        jp.any(marked_for_reset),
        lambda data: sim._reset(data, default_data, marked_for_reset),
        lambda data: data,
        data,
    )
    sim_time = data.core.steps / data.core.freq
    pos = data.states.pos[:, 0, :]
    vel = data.states.vel[:, 0, :]
    ang_vel = data.states.ang_vel[:, 0, :]
    success = _success(vel, ang_vel, action)
    terminated = _terminated(data.states.pos, success)
    truncated = _truncated(sim_time[..., 0], env.max_episode_time)
    # 4. construct obs & rewards
    steps = data.core.steps // (sim.freq // env.freq)
    # 5. Stop gradient flow through observations from marked-for-reset environments
    obs = jax.tree.map(
        lambda o: jp.where(
            marked_for_reset.reshape(-1, *([1] * (o.ndim - 1))), jax.lax.stop_gradient(o), o
        ),
        _obs(data),
    )
    env = env.replace(data=data, steps=steps, _marked_for_reset=terminated | truncated)
    return env, (
        obs,
        _reward(
            pos,
            vel,
            float(env.height_weight),
            float(env.vel_weight),
            float(env.vel_coef),
            float(env.floor_weight),
            float(env.floor_coef),
            float(env.ceiling_weight),
            float(env.ceiling_coef),
            float(env.fall_weight_below),
            float(env.fall_weight_above),
        ),
        terminated,
        truncated,
        {"success": success, "distance": jp.abs(pos[:, 2] - env.target_height)},
    )


def reset_rotor_velocities(
    data: SimData, default_data: SimData, mask: Array, *, idle_rotor_vel: float
) -> SimData:
    """Reset rotor velocities to the idle speed the drone falls with before the policy takes over."""
    rotor_vel = idle_rotor_vel * jp.ones(
        (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
    )
    data = data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor_vel))
    return data


def reset_randomization(data: SimData, default_data: SimData, mask: Array) -> SimData:
    """Randomize the z position, velocity and angular velocity of the drone.

    The drone spawns at a random height with a downward velocity and a random tumble.
    """
    key, z_key, q_key, ax_key, vel_key, av_ax_key, av_mag_key = jax.random.split(
        data.core.rng_key, 7
    )
    data = data.replace(core=data.core.replace(rng_key=key))
    n = data.core.n_worlds

    # Randomize z position
    z = jax.random.uniform(key=z_key, shape=(n,), minval=SPAWN_Z_MIN, maxval=SPAWN_Z_MAX)

    # Fully random orientation (uniform random unit quaternion)
    q_rng = jax.random.normal(key=q_key, shape=(n, 1, 4))
    q_rng = q_rng / jp.linalg.norm(q_rng, axis=-1, keepdims=True)

    # Velocity: downward with up to VEL_ANGLE_MAX angular disturbance
    qv_rng = jax.random.normal(key=ax_key, shape=(n, 1, 4))
    vel_scale = jax.random.uniform(key=vel_key, shape=(n, 1, 1), minval=0.0, maxval=VEL_MAX)
    vel_rng = (R.from_quat(qv_rng) ** (VEL_ANGLE_MAX / np.pi)).apply(
        jp.array([0.0, 0.0, -1.0])
    ) * vel_scale

    # Angular velocity: random direction, magnitude up to ANG_VEL_MAX
    av_axis = jax.random.normal(key=av_ax_key, shape=(n, 1, 3))
    av_axis = av_axis / jp.linalg.norm(av_axis, axis=-1, keepdims=True)
    av_mag = jax.random.uniform(key=av_mag_key, shape=(n, 1, 1), minval=0.0, maxval=ANG_VEL_MAX)
    ang_vel_rng = av_axis * av_mag

    pos = data.states.pos.at[...].set(0.0)
    pos = pos.at[..., 2].set(z[..., None])
    data = data.replace(
        states=leaf_replace(
            data.states, mask, pos=pos, quat=q_rng, vel=vel_rng, ang_vel=ang_vel_rng
        )
    )
    return data


def reset_randomization_x_start(
    data: SimData,
    default_data: SimData,
    mask: Array,
    *,
    pos_noise_range: float,
    pos_z_noise_std: float,
    pos_z_bounds: list[float],
    vel_noise_range: float,
    orient_noise_std: float,
    ang_vel_mag_noise_std: float,
    ang_vel_mag_max: float,
    ang_vel_dir_noise_std: float,
) -> SimData:
    """Sample the reset states around x_start.

    Position and velocity noise are standard normals cut at their range, except for the height,
    a normal around the predicted height cut at pos_z_bounds. The orientation is rotated by a
    Gaussian rotation vector in its own frame. The angular velocity is sampled in magnitude and
    direction: the magnitude from a normal around the predicted one cut to [0, ang_vel_mag_max],
    which keeps slow spins in the samples, the direction by rotating the predicted spin axis by a
    Gaussian rotation vector.
    """
    # One draw of each kind for all components, the reset runs in every step with a restart
    key, trunc_key, normal_key = jax.random.split(data.core.rng_key, 3)
    data = data.replace(core=data.core.replace(rng_key=key))
    n = data.core.n_worlds
    states = data.states

    z_low, z_high = pos_z_bounds
    z_pred = jp.clip(states.pos[..., 2:3], z_low, z_high)  # Inside, or no mass is left
    mag_pred = jp.linalg.norm(states.ang_vel, axis=-1, keepdims=True)
    axis_pred = states.ang_vel / jp.maximum(mag_pred, 1e-6)
    mag_pred = jp.clip(mag_pred, 0.0, ang_vel_mag_max)
    # Standard normals cut at: position (3), height (1), velocity (3), spin magnitude (1)
    ones = jp.ones((n, 1, 3))
    lower = jp.concatenate(
        [
            -pos_noise_range * ones,
            (z_low - z_pred) / pos_z_noise_std,
            -vel_noise_range * ones,
            -mag_pred / ang_vel_mag_noise_std,
        ],
        axis=-1,
    )
    upper = jp.concatenate(
        [
            pos_noise_range * ones,
            (z_high - z_pred) / pos_z_noise_std,
            vel_noise_range * ones,
            (ang_vel_mag_max - mag_pred) / ang_vel_mag_noise_std,
        ],
        axis=-1,
    )
    trunc = jax.random.truncated_normal(trunc_key, lower, upper, shape=(n, 1, 8))
    normal = jax.random.normal(normal_key, (n, 1, 6))  # Rotation vectors: orientation, spin axis

    pos = states.pos + trunc[..., 0:3]
    pos = pos.at[..., 2:3].set(z_pred + pos_z_noise_std * trunc[..., 3:4])
    vel = states.vel + trunc[..., 4:7]
    # The states hold unit quaternions already, normalizing them again only costs time
    quat_noise = R.from_rotvec(orient_noise_std * normal[..., 0:3])
    quat = (R(states.quat, normalize=False, copy=False) * quat_noise).as_quat()
    mag = mag_pred + ang_vel_mag_noise_std * trunc[..., 7:8]
    axis = R.from_rotvec(ang_vel_dir_noise_std * normal[..., 3:6]).apply(axis_pred)
    ang_vel = mag * axis

    data = data.replace(
        states=leaf_replace(data.states, mask, pos=pos, quat=quat, vel=vel, ang_vel=ang_vel)
    )
    return data


def _make_default_data(env: ThrowEnv, sim_default_data: SimData) -> SimData:
    """Return sim_default_data with states replaced by env.x_start_* (broadcast to all worlds)."""
    s = sim_default_data.states
    return sim_default_data.replace(
        states=s.replace(
            pos=jp.broadcast_to(env.x_start_pos, s.pos.shape),
            quat=jp.broadcast_to(env.x_start_quat, s.quat.shape),
            vel=jp.broadcast_to(env.x_start_vel, s.vel.shape),
            ang_vel=jp.broadcast_to(env.x_start_ang_vel, s.ang_vel.shape),
        )
    )


class ThrowEnv(struct.PyTreeNode):
    """Environment where the drone is thrown into the air and has to stabilise itself."""

    # Sim object for rendering
    sim: Sim = struct.field(pytree_node=False)

    # Constant environment parameters
    num_envs: int = struct.field(pytree_node=False)
    max_episode_time: float = struct.field(pytree_node=False)
    freq: int = struct.field(pytree_node=False)
    device: str = struct.field(pytree_node=False)
    single_action_space: spaces.Box = struct.field(pytree_node=False)
    action_space: spaces.Box = struct.field(pytree_node=False)
    single_observation_space: spaces.Dict = struct.field(pytree_node=False)
    observation_space: spaces.Dict = struct.field(pytree_node=False)
    n_substeps: int = struct.field(pytree_node=False)

    target_height: float = struct.field(pytree_node=False)
    height_weight: float = struct.field(pytree_node=False)
    vel_weight: float = struct.field(pytree_node=False)
    vel_coef: float = struct.field(pytree_node=False)
    floor_weight: float = struct.field(pytree_node=False)
    floor_coef: float = struct.field(pytree_node=False)
    ceiling_weight: float = struct.field(pytree_node=False)
    ceiling_coef: float = struct.field(pytree_node=False)
    fall_weight_below: float = struct.field(pytree_node=False)
    fall_weight_above: float = struct.field(pytree_node=False)
    control_type: Control | str = struct.field(pytree_node=False)

    # Variable simulation data
    data: SimData = struct.field(pytree_node=True)
    steps: Array = struct.field(pytree_node=True)
    _marked_for_reset: Array = struct.field(pytree_node=True)

    # x_start state for x_start-conditioned training (shape: (1, 1, dim) to broadcast)
    x_start_pos: Array = struct.field(pytree_node=True)
    x_start_quat: Array = struct.field(pytree_node=True)
    x_start_vel: Array = struct.field(pytree_node=True)
    x_start_ang_vel: Array = struct.field(pytree_node=True)

    # Functions
    reset: Callable = struct.field(pytree_node=False)
    step: Callable = struct.field(pytree_node=False)

    # Non-jittable functions
    def render(self, world: int = 0):
        """Render the simulation with a target zone rectangle at the target height."""
        z = self.target_height
        rectangle_corners = np.array(
            [[-2.0, -2.0, z], [2.0, -2.0, z], [2.0, 2.0, z], [-2.0, 2.0, z], [-2.0, -2.0, z]]
        )
        draw_line(
            self.sim, rectangle_corners, rgba=np.array([0, 1, 0, 1]), start_size=3.0, end_size=3.0
        )

        self.sim.data = self.data
        self.sim.render(
            world=world, cam_config={"lookat": (0.0, 0.0, 2.0), "distance": 5.0, "elevation": -10.0}
        )

    def close(self):
        """Close the underlying sim."""
        self.sim.close()

    @classmethod
    def create(
        cls,
        num_envs: int = 1,
        max_episode_time: float = 4.0,
        control_type: Control | str = Control.attitude,
        freq: int = 500,
        sim_freq: int = 500,
        attitude_freq: int = 500,
        force_torque_freq: int = 500,
        target_height: float = SUCCESS_Z,
        height_weight: float = 0.2,
        vel_weight: float = 1.0,
        vel_coef: float = 0.01,
        floor_weight: float = 1.0,
        floor_coef: float = 10.0,
        ceiling_weight: float = 1.0,
        ceiling_coef: float = 10.0,
        fall_weight_below: float = 0.0,
        fall_weight_above: float = 0.0,
        device: str = "cpu",
        reset_rotor: bool = True,
        use_x_start: bool = False,
        x_start_noise: dict | None = None,
    ) -> ThrowEnv:
        """Create a jittable drone environment without render support.

        Args:
            num_envs: Number of parallel environments.
            max_episode_time: Maximum episode time in seconds.
            control_type: Control interface the environment exposes.
            freq: Frequency of the simulation.
            sim_freq: Simulation frequency.
            attitude_freq: The frequency of the attitude controller.
            force_torque_freq: The frequency of the force/torque controller.
            device: Device to use for the simulation.
            target_height: The target height for the drone to stabilize at.
            vel_coef: Coefficient for the velocity term in the reward.
            height_weight: Slope of the linear height distance term in the reward.
            vel_weight: Weight for the velocity term in the reward.
            floor_weight: Weight for the floor proximity penalty.
            floor_coef: Coefficient for the floor proximity penalty.
            ceiling_weight: Weight for the ceiling proximity penalty.
            ceiling_coef: Coefficient for the ceiling proximity penalty.
            fall_weight_below: Weight of the linear downward speed penalty below the target height.
            fall_weight_above: Weight of the linear downward speed penalty above the target height.
            reset_rotor: Whether to reset rotor speeds to idle, the speed of the minimum thrust,
                on environment reset.
            use_x_start: If True, use x_start-conditioned domain randomization (noise around the
                x_start state). If False, use fully random throw initialization.
            x_start_noise: Noise of the x_start resets, the keyword arguments of
                reset_randomization_x_start. Required with use_x_start.

        Returns:
            An instance of ThrowEnv with jittable functions and data.
        """
        # Initialize the simulation
        jax_device = jax.devices(device)[0]
        sim = Sim(
            n_worlds=num_envs,
            n_drones=1,
            drone="cf21B_500",
            dynamics=Dynamics.first_principles,
            control=Control.attitude,
            device=device,
            freq=sim_freq,
            state_freq=100,
            attitude_freq=attitude_freq,
            force_torque_freq=force_torque_freq,
        )

        # Override reset randomization function
        if use_x_start:
            if x_start_noise is None:
                raise ValueError("The x_start resets need their noise ranges in x_start_noise")
            reset_fn = functools.partial(reset_randomization_x_start, **x_start_noise)
            append_fn(sim.reset_pipeline, reset_fn, name="reset_randomization_x_start")
        else:
            append_fn(sim.reset_pipeline, reset_randomization)
        params = load_params(Dynamics.first_principles, sim.drone)
        rpm2thrust = jp.asarray(params["rpm2thrust"])
        if reset_rotor:
            idle_rotor_vel = float(thrust2rotor_vel(params["thrust_min"], params["rpm2thrust"]))
            reset_fn = functools.partial(reset_rotor_velocities, idle_rotor_vel=idle_rotor_vel)
            append_fn(sim.reset_pipeline, reset_fn, name="reset_rotor_velocities")
        sim.build_reset_fn()
        # Remove all controllers, the rotors follow the commanded thrusts directly
        if control_type == "rotor_thrust":
            remove_fn(sim.step_pipeline, "attitude_controller")
            remove_fn(sim.step_pipeline, "force_torque_controller")
            sim.build_step_fn()

        # Prepare immutable constants
        single_action_space = create_action_space(sim.drone, control_type=control_type)
        action_space = batch_space(single_action_space, sim.n_worlds)
        single_observation_space = spaces.Dict(
            {
                "pos": spaces.Box(-np.inf, np.inf, shape=(1,)),
                "quat": spaces.Box(-np.inf, np.inf, shape=(4,)),
                "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )
        observation_space = batch_space(single_observation_space, sim.n_worlds)
        n_substeps = sim.freq // freq

        # Initialize reset mask and step count
        steps = jp.zeros((num_envs, 1), dtype=jp.int32, device=jax_device)
        _marked_for_reset = jp.zeros((num_envs,), dtype=jp.bool_, device=jax_device)

        # Default x_start taken from collected data
        x_start_pos = jax.device_put(jp.array([[[2.5808966, -0.29186085, 2.7325761]]]), jax_device)
        x_start_quat = jax.device_put(
            jp.array([[[0.01773736, 0.16663015, -0.8544407, -0.49178347]]]), jax_device
        )
        x_start_vel = jax.device_put(
            jp.array([[[-0.06380534, -0.13412078, -0.43453106]]]), jax_device
        )
        x_start_ang_vel = jax.device_put(
            jp.array([[[0.551279, -0.7471109, 3.4453685]]]), jax_device
        )

        return cls(
            sim=sim,
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            freq=freq,
            device=device,
            single_action_space=single_action_space,
            action_space=action_space,
            single_observation_space=single_observation_space,
            observation_space=observation_space,
            n_substeps=n_substeps,
            target_height=target_height,
            height_weight=height_weight,
            vel_weight=vel_weight,
            vel_coef=vel_coef,
            floor_weight=floor_weight,
            floor_coef=floor_coef,
            ceiling_weight=ceiling_weight,
            ceiling_coef=ceiling_coef,
            fall_weight_below=fall_weight_below,
            fall_weight_above=fall_weight_above,
            data=sim.data,
            steps=steps,
            _marked_for_reset=_marked_for_reset,
            x_start_pos=x_start_pos,
            x_start_quat=x_start_quat,
            x_start_vel=x_start_vel,
            x_start_ang_vel=x_start_ang_vel,
            reset=functools.partial(reset, sim=sim, device=jax_device),
            step=functools.partial(step, sim=sim, n_substeps=n_substeps, rpm2thrust=rpm2thrust),
            control_type=control_type,
        )
