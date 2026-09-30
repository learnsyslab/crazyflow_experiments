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
from crazyflow.sim.pipeline import append_fn, insert_fn_before, prepend_fn
from crazyflow.sim.visualize import draw_line, draw_points
from gymnasium import spaces
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space
from jax import Array

from crazyflow_experiments.control.trajectories import Figure8
from crazyflow_experiments.core.envs import domain_randomization as dr
from crazyflow_experiments.core.envs.drone_env import DroneEnv, create_action_space

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData


class FigureEightEnv(DroneEnv):
    """Figure Eight Environment."""

    autoreset_mode: AutoresetMode = struct.field(pytree_node=False)

    # Immutable figure-eight parameters
    trajectories: Array = struct.field(pytree_node=False)
    trajectory_vel: Array = struct.field(pytree_node=False)
    sample_offsets: Array = struct.field(pytree_node=False)
    randomization: dr.DynamicsRanges | None = struct.field(pytree_node=False, default=None)

    @property
    def target_pos(self) -> Array:
        """Current trajectory target position for all envs. Shape: (num_envs, 3)."""
        return self.trajectories[jp.arange(self.trajectories.shape[0])[:, None], self.steps][
            :, 0, :
        ]

    @property
    def target_vel(self) -> Array:
        """Current trajectory target velocity for all envs. Shape: (num_envs, 3)."""
        return self.trajectory_vel[jp.arange(self.trajectory_vel.shape[0])[:, None], self.steps][
            :, 0, :
        ]

    # Non-jittable functions
    def render(self, world: int = 0):
        """Override base class render to show figure-eight trajectory."""
        idx = self.steps + self.sample_offsets[None, ...] % self.trajectories.shape[1]
        next_trajectory = self.trajectories[jp.arange(self.trajectories.shape[0])[:, None], idx]
        trajectories = np.array(self.trajectories)
        next_trajectory = np.array(next_trajectory)
        draw_line(
            self.sim,
            trajectories[world, 0:-1:8, :],
            rgba=jp.array([1, 1, 1, 0.4]),
            start_size=2.0,
            end_size=2.0,
        )
        draw_line(
            self.sim,
            next_trajectory[world],
            rgba=jp.array([1, 0, 0, 1]),
            start_size=3.0,
            end_size=3.0,
        )
        draw_points(self.sim, next_trajectory[world], rgba=jp.array([1.0, 0, 0, 1]), size=0.01)
        current_target = self.trajectories[world, self.steps[world] % self.trajectories.shape[1]]
        draw_points(self.sim, current_target, rgba=jp.array([0.0, 1.0, 0.0, 0.4]), size=0.02)
        self.sim.data = self.data
        self.sim.render(world=world)

    @classmethod
    def create(
        cls,
        num_envs: int = 1,
        max_episode_time: float = 10.0,
        dynamics: Literal["so_rpy_rotor_drag", "first_principles"]
        | Dynamics = Dynamics.so_rpy_rotor_drag,
        control: Control | str = Control.default,
        drone: str = "cf21B_500",
        freq: int = 500,
        sim_freq: int = 500,
        state_freq: int = 100,
        attitude_freq: int = 500,
        force_torque_freq: int = 500,
        device: str = "cpu",
        n_samples: int = 10,
        trajectory_time: float = 10.0,
        samples_dt: float = 0.1,
        reset_rotor: bool = False,
        reset_randomization: Callable[[SimData, SimData, Array], SimData] | None = None,
        reset_velocity: bool = True,
        multi_start: bool = False,
        plane: Literal["xy", "xz"] = "xz",
        randomization: dr.DynamicsRanges | None = None,
    ) -> FigureEightEnv:
        """Create a jittable drone environment without render support.

        Args:
            num_envs: Number of parallel environments.
            max_episode_time: Maximum episode time in seconds.
            dynamics: Dynamics backend to use.
            control: Control interface to use.
            drone: Drone model of the environment.
            freq: Frequency of the simulation.
            sim_freq: Simulation frequency.
            state_freq: The frequency of the state controller.
            attitude_freq: The frequency of the attitude controller.
            force_torque_freq: The frequency of the force/torque controller.
            device: Device to use for the simulation.
            n_samples: Number of next trajectory points to sample for observations.
            trajectory_time: Total time for completing the figure-eight trajectory in seconds.
            samples_dt: Time between trajectory sample points in seconds.
            reset_rotor: Whether to reset rotor speeds on environment reset.
            reset_randomization: A function that randomizes the initial state of the simulation. If
                None, the default randomization for pos and vel is used.
            reset_velocity: Reset to reference velocity to avoid unfeasible acceleration.
            multi_start: Whether to start at multiple points along the trajectory.
            plane: Plane of the figure-8 trajectory. "xz" (default) keeps y=0 and oscillates in z.
                "xy" keeps z constant at 1.25 m and oscillates in y.
            randomization: Spread of the mass, inertia, rotor dynamics, thrust curve and external
                wrench drawn for every world at reset, and of the wrench noise added at every sim
                step. None keeps the nominal dynamics. Requires the first principles dynamics.

        Returns:
            An instance of FigureEightEnv with jittable functions and data.
        """
        if randomization is not None and Dynamics(dynamics) != Dynamics.first_principles:
            raise ValueError(f"Randomization requires first principles dynamics, got {dynamics}")
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
            state_freq=state_freq,
            attitude_freq=attitude_freq,
            force_torque_freq=force_torque_freq,
        )

        # Create the figure eight trajectory
        sample_offsets = jp.array(jp.arange(n_samples) * freq * samples_dt, dtype=int)
        # ensure enough steps for max episode time in addition to the future observation
        max_sample_idx = sample_offsets[-1] if n_samples > 0 else 0
        n_steps = int(max_episode_time * freq + max_sample_idx)
        sample_time = max_sample_idx / freq
        n_loops = max_episode_time / trajectory_time + sample_time / trajectory_time
        if plane not in ("xy", "xz"):
            raise ValueError(f"Invalid plane string: {plane}")
        # Each environment starts at its own point along the trajectory when multi_start is set
        offset = jp.linspace(0, trajectory_time, num_envs, endpoint=False)
        if not multi_start:
            offset = jp.zeros_like(offset)
        ts = jp.linspace(0, n_loops * trajectory_time, n_steps)[None, :] + offset[:, None]
        trajectory = Figure8(
            scaling=jp.array([2.0, 1.0, 0.0] if plane == "xy" else [2.0, 0.0, 1.0]),
            cycle_time=trajectory_time,
            duration=float(ts.max()),  # the trajectory clips beyond its duration
            origin=jp.array([0.0, 0.0, 1.25]),
            axes="xyz" if plane == "xy" else "xzy",
        )
        trajectories = trajectory.pos(ts)
        trajectory_vel = trajectory.vel(ts)

        # Assemble the reset and step pipelines from the domain randomization blocks
        if reset_randomization is None:
            # Keeps the historic draw: a fixed +0.1 m offset (both bounds equal), random velocity
            reset_randomization = dr.build_randomize_pos_vel_fn(
                pos_min=-0.1, pos_max=0.1, vel_min=-0.5, vel_max=0.5
            )
        append_fn(sim.reset_pipeline, reset_randomization, name="reset_randomization")
        if reset_rotor:
            # Spin up the rotors to help takeoff. The rotor state of the fitted models is thrust
            takeoff_rotor = 15900.0 if Dynamics(dynamics) == Dynamics.first_principles else 0.05
            append_fn(sim.reset_pipeline, dr.build_reset_rotor_fn(takeoff_rotor))
        if reset_velocity:
            append_fn(sim.reset_pipeline, dr.build_reset_velocity_fn(trajectory_vel[:, 0:1, :]))
        if randomization is not None:
            ranges = randomization
            sim.data = dr.add_wrench_plugin(sim.data)
            append_fn(sim.reset_pipeline, dr.build_randomize_mass_fn(ranges.mass))
            append_fn(sim.reset_pipeline, dr.build_randomize_inertia_fn(ranges.inertia))
            append_fn(
                sim.reset_pipeline, dr.build_randomize_rotor_dynamics_fn(ranges.rotor_dynamics)
            )
            append_fn(sim.reset_pipeline, dr.build_randomize_thrust_curve_fn(ranges.thrust_curve))
            append_fn(sim.reset_pipeline, dr.build_randomize_torque_curve_fn(ranges.torque_curve))
            append_fn(sim.reset_pipeline, dr.build_randomize_wrench_fn(ranges.force, ranges.torque))
            disturb = dr.build_disturb_wrench_fn(ranges.force_noise, ranges.torque_noise)
            insert_fn_before(sim.step_pipeline, "integration", disturb)
            delay_steps = round(ranges.action_delay * sim_freq)
            sim.data = dr.add_delay_plugin(sim.data, delay_steps)
            append_fn(sim.reset_pipeline, dr.build_randomize_delay_fn(delay_steps))
            delay = dr.build_delay_attitude_fn(sim_freq // freq)
            prepend_fn(sim.step_pipeline, delay, name="delay_attitude")
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

        # Set takeoff position and build default reset position
        takeoff_pos = trajectories[:, :1, :]
        sim.data = sim.data.replace(states=sim.data.states.replace(pos=takeoff_pos))
        sim.build_default_data()

        # Update observation space
        spec = {k: v for k, v in single_observation_space.items()}
        # use Python floats for infinity (compatible with gym spaces)
        spec["local_samples"] = spaces.Box(-float("inf"), float("inf"), shape=(3 * n_samples,))
        single_observation_space = spaces.Dict(spec)
        observation_space = batch_space(single_observation_space, sim.n_worlds)

        # Build jittable functions
        def _sanitize_action(action: Array, low: Array, high: Array) -> Array:
            action = jp.clip(action, low, high)
            return jp.array(action, device=jax_device).reshape((num_envs, 1, -1))

        def _aux_obs(
            trajectories: Array, steps: Array, pos: Array, sample_offsets: Array
        ) -> dict[str, Array]:
            """Static method version of obs for jitting."""
            # idx = jp.clip(steps + sample_offsets[None, ...], 0, trajectories.shape[1] - 1)
            # wrap around trajectory for cyclic behavior
            idx = (steps + sample_offsets[None, ...]) % trajectories.shape[1]
            dpos = trajectories[jp.arange(trajectories.shape[0])[:, None], idx] - pos
            local_samples = dpos.reshape(dpos.shape[0], dpos.shape[1] * dpos.shape[2])
            return local_samples

        def _obs(data: SimData) -> dict[str, Array]:
            obs = {
                "pos": data.states.pos[:, 0, :],
                "quat": data.states.quat[:, 0, :],
                "vel": data.states.vel[:, 0, :],
                "ang_vel": data.states.ang_vel[:, 0, :],
            }
            steps = data.core.steps // (sim.freq // freq)
            obs["local_samples"] = _aux_obs(trajectories, steps, data.states.pos, sample_offsets)
            return obs

        def _reset(
            env: FigureEightEnv, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[tuple[SimData, Array, Array], tuple[dict[str, Array], dict]]:
            data = env.data
            if seed is not None:
                rng_key = jax.device_put(jax.random.key(seed), jax_device)
                data = data.replace(core=data.core.replace(rng_key=rng_key))
            data = sim._reset(data, sim.default_data, None)
            _marked_for_reset = env._marked_for_reset.at[...].set(False)
            return env.replace(data=data, _marked_for_reset=_marked_for_reset), (_obs(data), {})

        def _reward(terminated: Array, pos: Array, goal: Array) -> Array:
            # distance to next trajectory point
            norm_distance = jp.linalg.norm(pos - goal, axis=-1)
            reward = jp.exp(-2.0 * norm_distance)
            reward += jp.where(terminated, -1.0, 0.0)
            return reward

        def _terminated(pos: Array) -> Array:
            lower_bounds = jp.array([-4.0, -4.0, 0.0])
            upper_bounds = jp.array([4.0, 4.0, 4.0])
            terminate = jp.any(
                (pos[:, 0, :] < lower_bounds) | (pos[:, 0, :] > upper_bounds), axis=-1
            )
            return terminate

        def _truncated(time: Array, max_episode_time: float) -> Array:
            return time >= max_episode_time

        def _done(terminated: Array, truncated: Array) -> Array:
            return terminated | truncated

        def _apply_action(data: SimData, action: Array, control: Control) -> SimData:
            low, high = action_space.low, action_space.high
            action = _sanitize_action(action, low, high)
            match control:
                case Control.state:
                    raise NotImplementedError("State control currently not supported")
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

        sim_step_fn = sim.build_step_fn()
        sim_reset_fn = sim.build_reset_fn()
        default_data = sim.build_default_data()

        def _step(
            env: FigureEightEnv, action: Array
        ) -> tuple[tuple[SimData, Array], tuple[Array, Array, Array, Array, dict]]:
            data = env.data
            # 1. apply action
            data = _apply_action(data, action)
            # 2. step sim for n_substeps
            data = sim_step_fn(data, n_substeps)
            # 3. compute termination on pre-reset state
            sim_time = data.core.steps / data.core.freq
            terminated = _terminated(data.states.pos)
            truncated = _truncated(sim_time[..., 0], max_episode_time)
            done = _done(terminated, truncated)
            # 4. compute reward and final_obs from pre-reset state.
            # SAME_STEP autoreset: done envs are reset within this step, so next_obs (returned
            # below) is the post-reset initial observation, not the true final state. The true
            # final obs is stored in info["final_obs"] so the replay buffer can bootstrap
            # correctly from it for truncated transitions without using the reset state.
            steps_pre = data.core.steps // (sim.freq // freq)
            pos_pre = data.states.pos[:, 0, :]
            goal_pre = trajectories[jp.arange(trajectories.shape[0])[:, None], steps_pre][:, 0, :]
            final_obs = _obs(data)
            final_rotor_vel = data.states.rotor_vel[:, 0, :]
            # 5. autoreset done envs (same step, no delayed reset)
            data = sim_reset_fn(data, default_data, done)
            steps = data.core.steps // (sim.freq // freq)
            info = {"final_obs": final_obs, "final_rotor_vel": final_rotor_vel}
            env = env.replace(data=data, steps=steps, _marked_for_reset=jp.zeros_like(done))
            reward = _reward(terminated, pos_pre, goal_pre)
            return env, (_obs(data), reward, terminated, truncated, info)

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
            trajectories=trajectories,
            trajectory_vel=trajectory_vel,
            sample_offsets=sample_offsets,
            randomization=randomization,
            data=sim.data,
            steps=steps,
            _marked_for_reset=_marked_for_reset,
            reset=jax.jit(_reset),
            step=jax.jit(_step),
        )
