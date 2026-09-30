"""Drone Environment Implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
from crazyflow.control import Control
from crazyflow.dynamics import load_params
from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import append_fn
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData


HOVER_RPM = 18967.0  # Rotor velocity that holds a cf2x in hover


def create_action_space(control_type: Control | str, drone: str) -> spaces.Box:
    """Select the appropriate action space for a given control type.

    Args:
        control_type: The desired control mode.
        drone: Drone model of the environment.

    Returns:
        The action space.
    """
    params = load_params(Dynamics.first_principles, drone)
    thrust_min, thrust_max = params["thrust_min"] * 4, params["thrust_max"] * 4
    match control_type:
        case Control.attitude:
            return spaces.Box(
                np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, thrust_min], dtype=np.float32),
                np.array([np.pi / 2, np.pi / 2, np.pi / 2, thrust_max], dtype=np.float32),
            )
        case Control.force_torque:
            return spaces.Box(
                np.array([thrust_min, -5e-4, -5e-4, -5e-4], dtype=np.float32),
                np.array([thrust_max, 5e-4, 5e-4, 5e-4], dtype=np.float32),
            )
        case "rotor_vel":
            rotor_vel_min, rotor_vel_max = (
                np.sqrt(params["thrust_min"] / params["rpm2thrust"][2]),
                np.sqrt(params["thrust_max"] / params["rpm2thrust"][2]),
            )
            return spaces.Box(rotor_vel_min, rotor_vel_max, shape=(4,))
        case _:
            raise ValueError(f"Invalid control type {control_type}")


class DroneEnv(struct.PyTreeNode):
    """Drone Environment.

    Pass the environment as an argument to jitted functions.

    Attributes:
        num_envs: Number of parallel environments.
        max_episode_steps: Maximum number of steps per episode.
        obs_space: Observation space of the environment.
        act_space: Action space of the environment.
        reset_fn: reset function.
        step_fn: step function.
        obs_fn: observation extraction function.
        data: Simulation data structure.
    """

    # Sim object for rendering
    sim: Sim = struct.field(pytree_node=False)
    # Constant environment parameters
    num_envs: int = struct.field(pytree_node=False)
    max_episode_time: float = struct.field(pytree_node=False)
    dynamics: Dynamics = struct.field(pytree_node=False)
    control: str = struct.field(pytree_node=False)
    drone: str = struct.field(pytree_node=False)
    freq: int = struct.field(pytree_node=False)
    device: str = struct.field(pytree_node=False)
    single_action_space: spaces.Box = struct.field(pytree_node=False)
    action_space: spaces.Box = struct.field(pytree_node=False)
    single_observation_space: spaces.Dict = struct.field(pytree_node=False)
    observation_space: spaces.Dict = struct.field(pytree_node=False)
    n_substeps: int = struct.field(pytree_node=False)

    # Variable simulation data
    data: SimData = struct.field(pytree_node=True)
    steps: Array = struct.field(pytree_node=True)
    _marked_for_reset: Array = struct.field(pytree_node=True)

    #  functions
    reset: Callable = struct.field(pytree_node=False)
    step: Callable = struct.field(pytree_node=False)

    # Non-jittable functions
    def render(self):
        """Sync current data into sim and call its render function."""
        self.sim.data = self.data
        self.sim.render(world=0)

    def close(self):
        """Close the underlying sim."""
        self.sim.close()

    @property
    def unwrapped(self) -> struct.PyTreeNode:
        """Return myself."""
        return self

    @classmethod
    def create(
        cls,
        num_envs: int = 1,
        max_episode_time: float = 10.0,
        dynamics: Literal["so_rpy_rotor_drag", "first_principles"]
        | Dynamics = Dynamics.first_principles,
        control: Control | str = Control.default,
        drone: str = "cf21B_500",
        freq: int = 500,
        sim_freq: int = 500,
        state_freq: int = 100,
        attitude_freq: int = 500,
        force_torque_freq: int = 500,
        device: str = "cpu",
        reset_randomization: Callable[[SimData, SimData, Array], SimData] | None = None,
    ) -> DroneEnv:
        """Create a jittable drone environment without render support.

        Args:
            num_envs: The number of environments to run in parallel.
            max_episode_time: The time horizon after which episodes are truncated (s).
            dynamics: The crazyflow dynamics simulation model.
            control: Control interface to use.
            drone: Drone model of the environment.
            freq: The frequency at which the environment is run.
            sim_freq: Simulation frequency.
            state_freq: The frequency of the state controller.
            attitude_freq: The frequency of the attitude controller.
            force_torque_freq: The frequency of the force/torque controller.
            device: The device of the environment and the simulation.
            reset_randomization: A function that randomizes the initial state of the simulation. If
                None, the default randomization for pos and vel is used.

        Returns:
            An instance of DroneEnv with jittable functions and data.

        Note:
            Override this create method to implement custom environments.
        """
        # Initialize the simulation
        jax_device = jax.devices(device)[0]
        sim = Sim(
            n_worlds=num_envs,
            n_drones=1,
            drone=drone,
            device=device,
            dynamics=dynamics,
            freq=sim_freq,
            state_freq=state_freq,
            attitude_freq=attitude_freq,
            force_torque_freq=force_torque_freq,
        )

        def _reset_randomization(data: SimData, default_data: SimData, mask: Array) -> SimData:
            """Randomize the initial position and velocity of the drones."""
            # Sample initial position
            shape = (data.core.n_worlds, data.core.n_drones, 3)
            pos_min = jp.array([-1.0, -1.0, 1.0])
            pos_max = jp.array([1.0, 1.0, 2.0])
            key, pos_key, vel_key = jax.random.split(data.core.rng_key, 3)
            data = data.replace(core=data.core.replace(rng_key=key))
            pos = jax.random.uniform(key=pos_key, shape=shape, minval=pos_min, maxval=pos_max)
            # Sample initial velocity
            vel = jax.random.uniform(key=vel_key, shape=shape, minval=-1.0, maxval=1.0)
            data = data.replace(states=leaf_replace(data.states, mask, pos=pos, vel=vel))
            return data

        if reset_randomization is None:
            reset_randomization = _reset_randomization
        append_fn(sim.reset_pipeline, reset_randomization, name="reset_randomization")
        sim.build_reset_fn()
        # Prepare immutable constants
        single_action_space = create_action_space(sim.control, sim.drone)
        action_space = batch_space(single_action_space, sim.n_worlds)
        single_observation_space = spaces.Dict(
            {
                "pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "quat": spaces.Box(-np.inf, np.inf, shape=(4,)),
                "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )
        observation_space = batch_space(single_observation_space, sim.n_worlds)
        n_substeps = sim.freq // freq

        # Build jittable functions
        def _sanitize_action(action: Array, low: Array, high: Array) -> Array:
            action = jp.clip(action, low, high)
            return jp.array(action, device=jax_device).reshape((num_envs, 1, -1))

        def _obs(data: SimData) -> dict[str, Array]:
            obs = {
                "pos": data.states.pos,
                "quat": data.states.quat,
                "vel": data.states.vel,
                "ang_vel": data.states.ang_vel,
            }
            return {k: v[:, 0, :] for k, v in obs.items()}

        def _reset(
            env: DroneEnv, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[tuple[SimData, Array, Array], tuple[dict[str, Array], dict]]:
            data = env.data
            _marked_for_reset = env._marked_for_reset
            if seed is not None:
                rng_key = jax.device_put(jax.random.key(seed), jax_device)
                data = data.replace(core=data.core.replace(rng_key=rng_key))
            data = sim._reset(data, sim.default_data, None)
            _marked_for_reset = env._marked_for_reset.at[...].set(False)
            return env.replace(data=data, _marked_for_reset=_marked_for_reset), (_obs(data), {})

        def _reward(data: SimData) -> Array:
            return 0.0 * jp.ones((num_envs,), dtype=jp.float32, device=jax_device)

        def _terminated(pos: Array) -> Array:
            return pos[:, 0, 2] < 0  # Terminate if the drone has crashed into the ground

        def _truncated(time: Array, max_episode_time: float) -> Array:
            return time >= max_episode_time

        def _done(terminated: Array, truncated: Array) -> Array:
            return terminated | truncated

        def _step(
            env: DroneEnv, action: Array
        ) -> tuple[tuple[SimData, Array], tuple[Array, Array, Array, Array, dict]]:
            data, _marked_for_reset = env.data, env._marked_for_reset
            # 1. apply action: only attitude control
            low, high = action_space.low, action_space.high
            action = _sanitize_action(action, low, high)
            data = data.replace(
                controls=data.controls.replace(
                    attitude=data.controls.attitude.replace(staged_cmd=action)
                )
            )
            # 2. step sim for n_substeps
            data = sim._step(data, n_substeps)
            # 3. handle autoreset & update mask
            data = sim._reset(data, sim.default_data, _marked_for_reset)
            sim_time = data.core.steps / data.core.freq
            terminated, truncated = (
                _terminated(data.states.pos),
                _truncated(sim_time[..., 0], max_episode_time),
            )
            _marked_for_reset = _done(terminated, truncated)
            # 4. construct obs & rewards
            steps = data.core.steps // (sim.freq // freq)
            # pos = data.states.pos[:, 0, :]
            # goal = trajectories[jp.arange(trajectories.shape[0])[:, None], steps][:, 0, :]
            # reward = _reward(terminated, pos, goal)

            return env.replace(data=data, steps=steps, _marked_for_reset=_marked_for_reset), (
                _obs(data),
                _reward(data),
                terminated,
                truncated,
                {},
            )

        # Initialize reset mask and step count
        steps = jp.zeros((num_envs, 1), dtype=jp.int32, device=jax_device)
        _marked_for_reset = jp.zeros((num_envs,), dtype=jp.bool_, device=jax_device)

        return cls(
            sim=sim,
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            dynamics=dynamics,
            control=control,
            drone=drone,
            freq=freq,
            device=device,
            single_action_space=single_action_space,
            action_space=action_space,
            single_observation_space=single_observation_space,
            observation_space=observation_space,
            n_substeps=n_substeps,
            data=sim.data,
            steps=steps,
            _marked_for_reset=_marked_for_reset,
            reset=jax.jit(_reset),
            step=jax.jit(_step),
        )


if __name__ == "__main__":
    import time

    """Test the jittable drone environment implementation."""
    # Create the jittable environment
    env = DroneEnv.create(
        num_envs=1024,
        max_episode_time=10.0,
        dynamics=Dynamics.so_rpy_rotor_drag,
        drone="cf21B_500",
        freq=500,
        device="gpu",
    )

    # Reset the environment
    env, (obs, info) = env.reset(env, seed=42)

    def step_once(env: DroneEnv, _) -> tuple[DroneEnv, tuple[Array, Array]]:
        """Single env step for lax.scan."""
        base_action = jp.array([0.0, 0.0, 0.0, 0.4], dtype=jp.float32)
        action = jp.broadcast_to(base_action, env.action_space.shape)  # (num_envs, act_dim)

        env, (next_obs, reward, terminated, truncated, info) = env.step(env, action)

        pos = env.data.states.pos[:, 0, :]  # (num_envs, 3)
        vel = env.data.states.vel[:, 0, :]  # (num_envs, 3)

        return env, (pos, vel)

    def rollout(env: DroneEnv, num_steps: int) -> tuple[DroneEnv, tuple[Array, Array]]:
        """Rollout for multiple steps using lax.scan."""
        env, (pos_traj, vel_traj) = jax.lax.scan(step_once, env, xs=None, length=num_steps)
        return env, (pos_traj, vel_traj)

    rollout_jit = jax.jit(rollout, static_argnames=("num_steps",))

    # Warm-up rollout
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 8)
    end_time = time.time()
    print(f"Warm-up rollout took {end_time - start_time:.4f} seconds")

    # After jitting
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 8)
    end_time = time.time()
    print(f"Jitted rollout took {end_time - start_time:.4f} seconds")
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 8)
    end_time = time.time()
    print(f"Jitted rollout took {end_time - start_time:.4f} seconds")

    print("\nPos trajectory shape:", pos_traj.shape)
    print("Vel trajectory shape:", vel_traj.shape)
