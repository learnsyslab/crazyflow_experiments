"""Drone Environment Implementation."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Literal

import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.pipeline import append_fn
from crazyflow.sim.visualize import draw_points
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space
from jax import Array

from crazyflow_experiments.core.envs.drone_env import HOVER_RPM, DroneEnv, create_action_space

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData


class ReachPosEnv(DroneEnv):
    """Reach Position Environment.

    Pass the environment as an argument to jitted functions.

    Attributes:
        num_envs: Number of parallel environments.
        max_episode_steps: Maximum number of steps per episode.
        obs_space: Observation space of the figure-eight environment.
        act_space: Action space of the figure-eight environment.
        reset_fn: reset function.
        step_fn: step function.
        obs_fn: observation extraction function.
        data: Simulation data structure.
    """

    autoreset_mode: AutoresetMode = struct.field(pytree_node=False)

    # Reach position specific attributes
    goal_pos: Array = struct.field(pytree_node=True)

    # Non-jittable functions
    def render(self, world: int = 0):
        """Override base class render to show reach position."""
        draw_points(
            self.sim, self.goal_pos[None, world], rgba=jp.array([1.0, 0, 0, 1.0]), size=0.01
        )
        self.sim.data = self.data
        self.sim.render(world=world)

    @classmethod
    def create(
        cls,
        num_envs: int = 1,
        max_episode_time: float = 5.0,
        dynamics: Literal["so_rpy_rotor_drag", "first_principles"]
        | Dynamics = Dynamics.first_principles,
        control: Control | str = Control.default,
        drone: str = "cf21B_500",
        freq: int = 500,
        sim_freq: int = 500,
        device: str = "cpu",
        reset_rotor: bool = False,
        pos_min: Array = jp.array([-1.0, -1.0, 1.0]),
        pos_max: Array = jp.array([1.0, 1.0, 2.0]),
        goal_pmin: Array = jp.array([-1.0, -1.0, 0.5]),
        goal_pmax: Array = jp.array([1.0, 1.0, 1.5]),
        ang_vel_min: Array = jp.zeros(3),
        ang_vel_max: Array = jp.zeros(3),
        vel_min: float = -1.0,
        vel_max: float = 1.0,
    ) -> ReachPosEnv:
        """Create a jittable drone environment without render support.

        Args:
            num_envs: Number of parallel environments.
            max_episode_time: Maximum episode time in seconds.
            dynamics: Dynamics backend to use.
            control: Control interface.
            drone: Drone model of the environment.
            freq: Frequency of the simulation.
            sim_freq: Simulation frequency.
            device: Device to use for the simulation.
            n_samples: Number of next trajectory points to sample for observations.
            trajectory_time: Total time for completing the figure-eight trajectory in seconds.
            samples_dt: Time between trajectory sample points in seconds.
            reset_rotor: Whether to reset rotor speeds on environment reset.
            pos_min: Minimum position for randomization on reset.
            pos_max: Maximum position for randomization on reset.
            goal_pmin: Minimum goal position for random goal sampling.
            goal_pmax: Maximum goal position for random goal sampling.
            ang_vel_min: Minimum initial angular velocity for randomization.
            ang_vel_max: Maximum initial angular velocity for randomization.
            vel_min: Minimum velocity for randomization on reset.
            vel_max: Maximum velocity for randomization on reset.

        Returns:
            An instance of ReachPosEnv with jittable functions and data.
        """
        # Initialize the simulation
        jax_device = jax.devices(device)[0]
        sim = Sim(
            n_worlds=num_envs,
            n_drones=1,
            drone=drone,
            dynamics=dynamics,
            control=control if control in [c.value for c in Control] else Control.default,
            device=device,
            freq=sim_freq,
        )

        # Override reset randomization function
        def build_reset_rotor_fn(dynamics: str) -> Callable[[SimData, SimData, Array], SimData]:
            """Reset rotor."""

            # Spin up rotors to help takeoff
            def _reset_rotor_so_rpy(data: SimData, default_data: SimData, mask: Array) -> SimData:
                rotor_vel = 0.05 * jp.ones(
                    (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
                )
                data = data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor_vel))
                return data

            def _reset_rotor_first_principles(
                data: SimData, default_data: SimData, mask: Array
            ) -> SimData:
                rotor_vel = HOVER_RPM * jp.ones(
                    (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
                )
                data = data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor_vel))
                return data

            def _no_reset_rotor(data: SimData, default_data: SimData, mask: Array) -> SimData:
                return data

            match dynamics:
                case "first_principles":
                    return _reset_rotor_first_principles
                case "so_rpy" | "so_rpy_rotor" | "so_rpy_rotor_drag":
                    return _reset_rotor_so_rpy
                case "no_reset_rotor":
                    return _no_reset_rotor

        def _reset_randomization(
            data: SimData,
            default_data: SimData,
            mask: Array,
            pmin: Array,
            pmax: Array,
            vmin: float,
            vmax: float,
            wmin: Array,
            wmax: Array,
        ) -> SimData:
            shape = (data.core.n_worlds, data.core.n_drones, 3)
            key, pos_key, vel_key, ang_vel_key = jax.random.split(data.core.rng_key, 4)
            data = data.replace(core=data.core.replace(rng_key=key))
            pos = jax.random.uniform(key=pos_key, shape=shape, minval=pmin, maxval=pmax)
            vel = jax.random.uniform(key=vel_key, shape=shape, minval=vmin, maxval=vmax)
            ang_vel = jax.random.uniform(key=ang_vel_key, shape=shape, minval=wmin, maxval=wmax)
            data = data.replace(
                states=leaf_replace(data.states, mask, pos=pos, vel=vel, ang_vel=ang_vel)
            )
            return data

        reset_rotor_randomization = build_reset_rotor_fn(
            dynamics if reset_rotor else "no_reset_rotor"
        )

        reset_randomization = functools.partial(
            _reset_randomization,
            pmin=pos_min,
            pmax=pos_max,
            vmin=vel_min,
            vmax=vel_max,
            wmin=ang_vel_min,
            wmax=ang_vel_max,
        )
        append_fn(sim.reset_pipeline, reset_randomization, name="reset_randomization")
        append_fn(sim.reset_pipeline, reset_rotor_randomization, name="reset_rotor_randomization")
        sim.build_reset_fn()

        # Prepare immutable constants
        single_action_space = create_action_space(control, sim.drone)
        action_space = batch_space(single_action_space, sim.n_worlds)
        single_observation_space = spaces.Dict(
            {
                "pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "quat": spaces.Box(-np.inf, np.inf, shape=(4,)),
                "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )
        n_substeps = sim.freq // freq

        # Update observation space
        # spec = {k: v for k, v in single_observation_space.items()}
        # spec["difference_to_goal"] = spaces.Box(-np.inf, np.inf, shape=(3,))
        # single_observation_space = spaces.Dict(spec)
        observation_space = batch_space(single_observation_space, sim.n_worlds)

        # Build jittable functions
        def _sanitize_action_STE(action: Array, low: Array, high: Array) -> Array:
            action_clipped = jp.clip(action, low, high)
            action = action + jax.lax.stop_gradient(action_clipped - action)
            return jp.array(action, device=jax_device).reshape((num_envs, 1, -1))

        def _obs(goal_pos: Array, data: SimData) -> dict[str, Array]:
            obs = {
                "pos": data.states.pos[:, 0, :],
                "quat": data.states.quat[:, 0, :],
                "vel": data.states.vel[:, 0, :],
                "ang_vel": data.states.ang_vel[:, 0, :],
            }
            # obs["difference_to_goal"] = goal_pos - data.states.pos[:, 0, :] # legacy approach
            obs["pos"] = data.states.pos[:, 0, :] - goal_pos  # agent only sees relative position
            return obs

        def _sample_goal(key: Array, goal: Array, mask: Array | None) -> Array:
            new_goal = jax.random.uniform(key, shape=goal.shape, minval=goal_pmin, maxval=goal_pmax)
            if mask is not None:
                new_goal = jp.where(mask[..., None], new_goal, goal)
            return new_goal

        def _reset(
            env: ReachPosEnv, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[tuple[SimData, Array, Array], tuple[dict[str, Array], dict]]:
            data = env.data
            if seed is not None:
                rng_key = jax.device_put(jax.random.key(seed), jax_device)
                data = data.replace(core=data.core.replace(rng_key=rng_key))
            # 1. re-sample goal position
            rng_key, subkey = jax.random.split(data.core.rng_key)
            data = data.replace(core=data.core.replace(rng_key=rng_key))
            goal_pos = _sample_goal(subkey, env.goal_pos, None)
            # 2. reset sim
            data = sim._reset(data, sim.default_data, None)
            _marked_for_reset = env._marked_for_reset.at[...].set(False)
            env = env.replace(data=data, _marked_for_reset=_marked_for_reset, goal_pos=goal_pos)
            return env, (_obs(goal_pos, data), {"success": jp.zeros((num_envs,), dtype=bool)})

        def _reward(terminated: Array, pos: Array, goal: Array, success: Array) -> Array:
            # distance to next trajectory point
            norm_distance = jp.linalg.norm(pos - goal, axis=-1)
            reward = jp.exp(-6.0 * norm_distance)
            reward = jp.where(success, 100.0, reward)
            reward = jp.where(terminated & ~success, -1.0, reward)
            return reward

        def _terminated(env: ReachPosEnv, sim_data: SimData) -> tuple[Array, Array]:
            pos = sim_data.states.pos[:, 0, :]
            lower_bounds = jp.array([-4.0, -4.0, 0.0])
            upper_bounds = jp.array([4.0, 4.0, 4.0])
            out_of_bounds = jp.any((pos < lower_bounds) | (pos > upper_bounds), axis=-1)
            at_goal = jp.linalg.norm(pos - env.goal_pos, axis=-1) < 0.05
            at_rest = jp.linalg.norm(sim_data.states.vel[:, 0, :], axis=-1) < 0.05
            return out_of_bounds | (at_goal & at_rest), (at_goal & at_rest)

        def _truncated(time: Array, max_episode_time: float) -> Array:
            return time >= max_episode_time

        def _done(terminated: Array, truncated: Array) -> Array:
            return terminated | truncated

        def _apply_action(data: SimData, action: Array, control: Control) -> SimData:
            low, high = action_space.low, action_space.high
            action = _sanitize_action_STE(action, low, high)
            match control:
                case Control.attitude:
                    data = data.replace(
                        controls=data.controls.replace(
                            attitude=data.controls.attitude.replace(staged_cmd=action)
                        )
                    )
                case Control.force_torque:
                    data = data.replace(
                        controls=data.controls.replace(
                            force_torque=data.controls.force_torque.replace(staged_cmd=action)
                        )
                    )
                case "rotor_vel":
                    data = data.replace(controls=data.controls.replace(rotor_vel=action))
                case _:
                    raise ValueError(f"Invalid control type {control}")
            return data

        _apply_action = functools.partial(_apply_action, control=control)

        def _step(
            env: ReachPosEnv, action: Array
        ) -> tuple[tuple[SimData, Array], tuple[Array, Array, Array, Array, dict]]:
            data = env.data
            # 1. apply action
            data = _apply_action(data, action)
            # 2. step sim for n_substeps
            data = sim._step(data, n_substeps)
            # 3. compute termination on pre-reset state
            sim_time = data.core.steps / data.core.freq
            terminated, success = _terminated(env, data)
            truncated = _truncated(sim_time[..., 0], max_episode_time)
            done = _done(terminated, truncated)
            # 4. compute reward and final_obs from pre-reset state.
            # SAME_STEP autoreset: done envs are reset within this step, so next_obs (returned
            # below) is the post-reset initial observation, not the true final state. The true
            # final obs is stored in info["final_obs"] so the replay buffer can bootstrap
            # correctly from it for truncated transitions without using the reset state.
            pos_pre = data.states.pos[:, 0, :]
            final_obs = _obs(env.goal_pos, data)
            final_rotor_vel = data.states.rotor_vel[:, 0, :]
            episode_steps = (data.core.steps[:, 0] // (sim.freq // freq)).astype(jp.int32)
            reward = _reward(terminated, pos_pre, env.goal_pos, success)
            # 5. autoreset done envs (same step, no delayed reset): resample goal, reset sim
            rng_key, subkey = jax.random.split(data.core.rng_key)
            data = data.replace(core=data.core.replace(rng_key=rng_key))
            goal_pos = _sample_goal(subkey, env.goal_pos, done)
            data = sim._reset(data, sim.default_data, done)
            steps = data.core.steps // (sim.freq // freq)
            obs = _obs(goal_pos, data)
            env = env.replace(
                data=data, steps=steps, _marked_for_reset=jp.zeros_like(done), goal_pos=goal_pos
            )
            info = {
                "final_obs": final_obs,
                "final_rotor_vel": final_rotor_vel,
                "success": success,
                "episode_steps": episode_steps,
            }
            return env, (obs, reward, terminated, truncated, info)

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
            autoreset_mode=AutoresetMode.SAME_STEP,
            goal_pos=jp.zeros((sim.n_worlds, 3), dtype=jp.float32, device=jax_device),
            data=sim.data,
            steps=steps,
            _marked_for_reset=_marked_for_reset,
            reset=jax.jit(_reset),
            step=jax.jit(_step),
        )
