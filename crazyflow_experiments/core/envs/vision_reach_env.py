"""Vision reach environment."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.struct as struct
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import splax
from crazyflow.control import Control
from crazyflow.control import load_params as load_control_params
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.sim import Sim
from crazyflow.sim.functional import attitude_control, state_control
from crazyflow.sim.pipeline import append_fn
from crazyflow.sim.sensors.splat import build_render_splat_fn
from crazyflow.sim.splat import SPLATS_KEY, attach_splats
from crazyflow.sim.visualize import draw_points
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space
from jax import Array
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.core.envs import domain_randomization as dr
from crazyflow_experiments.core.envs.augmentation import CameraNoise, build_augment, build_zoom
from crazyflow_experiments.core.envs.drone_env import DroneEnv
from crazyflow_experiments.core.envs.wrappers import Wrapper

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Callable, Sequence

    from crazyflow.sim.data import SimData


def _splat_pool(attached: Any, variants: Sequence[Path]) -> Any:
    """Stack the attached reconstruction and every variant along a leading axis."""
    size, orders = attached.means.shape[0], attached.sh_colors.shape[1]
    drawn = [attached]
    for path in variants:
        means, log_scales, quats, sh, opacities = splax.io.load_ply(path)
        pad = size - means.shape[0]
        assert pad >= 0, f"{path.stem} has more gaussians than the attached scene"
        assert sh.shape[1] <= orders, f"{path.stem} carries harmonics of a higher order"
        blank = jp.zeros((pad, 3))
        sh = jp.concatenate([sh, jp.zeros((sh.shape[0], orders - sh.shape[1], 3))], axis=1)
        drawn.append(
            dataclasses.replace(
                attached,
                means=jp.concatenate([means, blank]),
                log_scales=jp.concatenate([log_scales, blank]),
                quats=jp.concatenate([quats, jp.tile(jp.array([0.0, 0.0, 0.0, 1.0]), (pad, 1))]),
                sh_colors=jp.concatenate([sh, jp.zeros((pad, orders, 3))]),
                logit_opacities=jp.concatenate([opacities, jp.full((pad,), -20.0)]),
            )
        )
    return jax.tree.map(lambda *leaves: jp.concatenate(leaves), *drawn)


def setpoints(pos: Array, rot: R, action: Array) -> tuple[Array, Array]:
    """Turn a policy action into the setpoints the onboard state interface flies to.

    Args:
        pos: Drone position in the world frame as (..., 3).
        rot: Drone orientation.
        action: Policy output in [-1, 1] as (..., 4), a step in the body frame and a yaw step.

    Returns:
        The position setpoint in the world frame and the yaw setpoint, both absolute.
    """
    step = VisionReachEnv.DELTA_MAX * action[..., :3]
    return pos + rot.apply(step), rot.as_euler("xyz")[..., 2] + VisionReachEnv.DYAW_MAX * action[
        ..., 3
    ]


def mount_camera(sim: Sim, fovy: float):
    """Give the onboard camera its position, orientation and field of view."""
    cameras = [c for c in sim.spec.cameras if c.name.startswith("fpv_cam")]
    assert len(cameras) == 1, "the drone model has no fpv_cam to render from"
    pointed = (
        R.from_euler("xyz", VisionReachEnv.CAM_ROT, degrees=True) * VisionReachEnv.LOOK_FORWARD
    )
    cameras[0].pos = VisionReachEnv.CAM_POS
    cameras[0].quat = pointed.as_quat(scalar_first=True)
    cameras[0].alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    cameras[0].fovy = fovy
    sim.build_mjx()


def attitudes(yaw: Array, action: Array) -> tuple[Array, Array]:
    """Turn a policy action into attitude setpoints, turning from the drone's own yaw."""
    new_yaw = yaw + VisionReachEnv.DYAW_MAX * action[..., 2]
    yaw = jp.arctan2(jp.sin(new_yaw), jp.cos(new_yaw))
    tilt = VisionReachEnv.TILT_MAX
    rpy = jp.stack([tilt * action[..., 0], tilt * action[..., 1], yaw], axis=-1)
    low, high = VisionReachEnv.THRUST_MIN, VisionReachEnv.THRUST_MAX
    return rpy, low + (high - low) * (action[..., 3] + 1.0) / 2.0


def set_inertial(data: SimData, mass: float) -> SimData:
    """Give the simulated drone and its onboard controller the drone's measured mass in kg."""
    scale = mass / VisionReachEnv.PARAMS["mass"]
    params = data.params.replace(
        mass=jp.full_like(data.params.mass, mass),
        J=data.params.J * scale,
        J_inv=data.params.J_inv / scale,
    )
    controls = data.controls
    if controls.state is not None:
        state = controls.state  # We also need to change the mass in the state interface
        controller = state.params | {
            "mass": jp.full_like(state.params["mass"], mass),
            "mass_thrust": jp.full_like(state.params["mass_thrust"], VisionReachEnv.MASS_THRUST),
        }
        controls = controls.replace(state=state.replace(params=controller))
    return data.replace(params=params, controls=controls)


def _mass_thrust(drone: str) -> int:
    control_params = load_control_params("mellinger", drone)["core"]
    return round(control_params["pwm_max"] / (4 * control_params["thrust_max"]))


class VisionReachEnv(DroneEnv):
    """Vision Reach Environment with randomly sampled goals."""

    autoreset_mode: AutoresetMode = struct.field(pytree_node=False)

    draw_scene: Callable = struct.field(pytree_node=False)
    goal_pos: Array = struct.field(pytree_node=True)
    domain: dr.Domain = struct.field(pytree_node=True)
    settled: Array = struct.field(pytree_node=True)

    # Unannotated class constants.
    SETTLE_TIME = 0.2  # s the drone holds the goal at rest before the episode ends
    RENDER_SCALE = 1.25  # Increased render scale for FOV randomization
    FREQ = 50  # Control frequency in Hz
    SIM_FREQ = 500  # Dynamics frequency in Hz, at which the onboard controllers run as well

    # Camera parameters
    RESOLUTION = (64, 64)
    LUMINANCE = jp.array([0.299, 0.587, 0.114])
    CAM_POS = (0.0, 0.0, 0.04)
    CAM_ROT = (0.8, -2.35, 3.7)  # xyz Euler in deg
    CAM_FOVY = 70.0  # FOV of the camera in degrees
    # MuJoCo camera convention
    LOOK_FORWARD = R.from_matrix(np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    # Drone and action constants
    DRONE = "cf21B_500"
    MASS = 0.070  # The crazyflie with cameras is heavier
    PARAMS = load_params(Dynamics.first_principles, DRONE)
    MASS_THRUST = _mass_thrust(DRONE)
    DELTA_MAX = 0.3  # maximum position increment per step in meters
    DYAW_MAX = 0.11  # maximum heading increment per step in radians
    TILT_MAX = 0.6  # maximum roll and pitch in radians
    THRUST_MAX = 4 * PARAMS["thrust_max"]  # maximum collective thrust in N
    THRUST_MIN = 2 * MASS * 9.81 - THRUST_MAX  # mirror of the maximum about hover

    # Non-jittable functions
    def render(self):
        """Override base class render to show the goal position."""
        draw_points(self.sim, self.goal_pos[None, 0], rgba=jp.array([1.0, 0, 0, 1.0]), size=0.01)
        self.sim.data = self.data
        self.sim.render()

    @classmethod
    def create(
        cls,
        scene: Path,
        variants: Sequence[Path] = (),
        n_envs: int = 1,
        max_episode_time: float = 8.0,
        control: Control | str = Control.state,
        ranges: dr.Ranges = dr.Ranges(),
        noise: CameraNoise = CameraNoise(),
        goal_radius: float = 0.2,
        rest_speed: float = 0.1,
        resolution: tuple[int, int] = RESOLUTION,
        bounds_low: tuple[float, float, float] = (-1.5, -1.2, 0.1),
        bounds_high: tuple[float, float, float] = (2.0, 1.4, 2.3),
        goal_low: tuple[float, float, float] = (-1.3, -1.0, 0.8),
        goal_high: tuple[float, float, float] = (1.8, 1.2, 1.6),
        spawn_low: tuple[float, float, float] = (-1.3, -1.0, 0.2),
        spawn_high: tuple[float, float, float] = (1.8, 1.2, 2.2),
    ) -> VisionReachEnv:
        """Create a jittable vision reach environment without render support.

        Args:
            scene: Path to the scene splat the observations are rendered from.
            variants: Further reconstructions of the same scene to draw from.
            n_envs: Number of parallel environments.
            max_episode_time: Maximum episode time in seconds.
            control: Control interface the actions are sent through.
            ranges: Spread of every randomized quantity.
            noise: Magnitude of every camera effect.
            goal_radius: Distance in m counted as reaching the goal.
            rest_speed: Speed in m/s below which the drone counts as having settled on it.
            resolution: Policy image resolution, as (width, height).
            bounds_low: Lower corner of the flyable space.
            bounds_high: Upper corner of the flyable space.
            goal_low: Lower corner of the box goals are drawn from.
            goal_high: Upper corner of the goal box.
            spawn_low: Lower corner of the box the drone spawns in.
            spawn_high: Upper corner of the spawn box.
        """
        jax_device = jax.devices("gpu")[0]
        sim = Sim(
            n_worlds=n_envs,
            drone=cls.DRONE,
            control=control,
            freq=cls.SIM_FREQ,
            attitude_freq=cls.SIM_FREQ,
            body_rate_freq=cls.SIM_FREQ,
            force_torque_freq=cls.SIM_FREQ,
            device="gpu",
        )
        render_fovy = cls.CAM_FOVY + ranges.cam_fovy_range
        mount_camera(sim, render_fovy)
        attach_splats(sim, scene=scene)
        pool_key = "splat_pool"
        pool = _splat_pool(sim.data.plugins[SPLATS_KEY], variants)
        sim.data = sim.data.replace(plugins=sim.data.plugins | {pool_key: pool})
        sim.data = set_inertial(sim.data, cls.MASS)  # The camera makes the drone heavier
        sim.build_default_data()

        floor, ceiling = bounds_low[2], bounds_high[2]
        settle_steps = round(cls.SETTLE_TIME * cls.FREQ)
        lower = jp.array(bounds_low[:2], jp.float32, device=jax_device)
        upper = jp.array(bounds_high[:2], jp.float32, device=jax_device)
        goal_low = jp.array(goal_low, jp.float32, device=jax_device)
        goal_high = jp.array(goal_high, jp.float32, device=jax_device)
        spawn_low = jp.array(spawn_low, jp.float32, device=jax_device)
        spawn_high = jp.array(spawn_high, jp.float32, device=jax_device)
        # Rotor speed holding hover thrust, so an episode does not open with the drone dropping
        polynomial = np.asarray(cls.PARAMS["rpm2thrust"], dtype=np.float64)[::-1].copy()
        polynomial[-1] -= cls.MASS * 9.81 / 4
        hover_rpm = float(np.roots(polynomial).max())

        def _randomize_mass_inertia(data: SimData, default_data: SimData, mask: Array) -> SimData:
            key, inertia_key = jax.random.split(data.core.rng_key)
            data = dr.randomize_mass_inertia(data, default_data, mask, inertia_key, ranges)
            return data.replace(core=data.core.replace(rng_key=key))

        def _randomize_position(data: SimData, _: SimData, mask: Array | None) -> SimData:
            shape = (data.core.n_worlds, data.core.n_drones)
            key, pos_key, yaw_key = jax.random.split(data.core.rng_key, 3)
            data = data.replace(core=data.core.replace(rng_key=key))
            pos = jax.random.uniform(pos_key, (*shape, 3), minval=spawn_low, maxval=spawn_high)
            yaw = jax.random.uniform(yaw_key, shape, minval=-jp.pi, maxval=jp.pi)
            quat = R.from_euler("z", yaw[..., None]).as_quat()
            return data.replace(
                states=leaf_replace(
                    data.states,
                    mask,
                    pos=pos,
                    quat=quat.reshape(*shape, 4),
                    vel=jp.zeros_like(pos),
                    ang_vel=jp.zeros_like(pos),
                    rotor_vel=jp.full((*shape, 4), hover_rpm),
                )
            )

        append_fn(sim.reset_pipeline, _randomize_mass_inertia, name="randomize_mass_inertia")
        append_fn(sim.reset_pipeline, _randomize_position, name="randomize_position")
        sim_reset = sim.build_reset_fn()
        sim_step = sim.build_step_fn()
        default_data = sim.default_data

        def _draw_scene(env: VisionReachEnv, key: Array) -> VisionReachEnv:
            data = env.unwrapped.data
            pool = data.plugins[pool_key]
            size = data.plugins[SPLATS_KEY].means.shape[0]
            start = jax.random.randint(key, (), 0, pool.means.shape[0] // size) * size
            drawn = jax.tree.map(lambda leaf: jax.lax.dynamic_slice_in_dim(leaf, start, size), pool)
            return Wrapper.recursive_replace(
                env, data=data.replace(plugins=data.plugins | {SPLATS_KEY: drawn})
            )

        width, height = resolution
        single_action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        action_space = batch_space(single_action_space, sim.n_worlds)
        single_observation_space = spaces.Dict(
            {
                "frame": spaces.Box(0.0, 1.0, shape=(1, height, width)),
                "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "acc": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "goal": spaces.Box(-np.inf, np.inf, shape=(3,)),  # In  world frame
            }
        )
        observation_space = batch_space(single_observation_space, sim.n_worlds)

        n_substeps = sim.freq // cls.FREQ
        render_resolution = (round(cls.RENDER_SCALE * width), round(cls.RENDER_SCALE * height))
        render_rgb = build_render_splat_fn(sim, resolution=render_resolution)
        zoom = build_zoom(render_resolution, resolution, render_fovy)
        augment = build_augment(resolution, cls.CAM_FOVY, noise)

        def _frame(data: SimData, domain: dr.Domain, key: Array) -> Array:
            """Grayscale image of the onboard camera, seen through its randomized mounting."""
            rot = R.from_quat(data.states.quat[:, 0, :])
            # The model carries the mount, so only the per world jitter goes through the pose
            pos = data.states.pos[:, 0, :] + rot.apply(domain.cam_offset)
            quat = (rot * R.from_quat(domain.cam_rot)).as_quat()
            states = data.states.replace(pos=pos[:, None, :], quat=quat[:, None, :])
            render = render_rgb(data.replace(states=states))[:, 0] @ cls.LUMINANCE
            image = zoom(render, cls.CAM_FOVY + domain.cam_fovy)
            body_vel = rot.apply(data.states.vel[:, 0, :], inverse=True)
            ang_vel = data.states.ang_vel[:, 0, :]
            keys = jax.random.split(key, n_envs)
            return augment(image, domain.photometry, ang_vel, body_vel, keys)[:, None]

        def _imu(data: SimData, imu_bias: Array, key: Array) -> tuple[Array, Array]:
            """Gyroscope and accelerometer readings with a drifting bias and white noise."""
            gyro_key, accel_key = jax.random.split(key)
            rot = R.from_quat(data.states.quat[:, 0, :])
            specific_force = rot.apply(
                data.states_deriv.acc[:, 0, :] - data.params.gravity_vec, inverse=True
            )
            gyro = data.states.ang_vel[:, 0, :] + imu_bias[:, :3]
            gyro = gyro + jp.asarray(ranges.gyro_noise) * jax.random.normal(gyro_key, (n_envs, 3))
            accel = specific_force + imu_bias[:, 3:]
            accel_noise = jp.asarray(ranges.accel_noise)
            accel = accel + accel_noise * jax.random.normal(accel_key, (n_envs, 3))
            return gyro, accel

        def _obs(frame: Array, ang_vel: Array, acc: Array, goal: Array) -> dict[str, Array]:
            return {"frame": frame, "ang_vel": ang_vel, "acc": acc, "goal": goal}

        def _apply_action(data: SimData, action: Array) -> SimData:
            """Turn the action into a command for the onboard controller."""
            action = jp.clip(action, -1.0, 1.0)
            rot = R.from_quat(data.states.quat[:, 0, :])
            if control == Control.attitude:
                rpy, thrust = attitudes(rot.as_euler("xyz")[:, 2], action)
                cmd = jp.concatenate([rpy, thrust[:, None]], axis=-1)[:, None, :]
                return attitude_control(data, cmd)
            pos = data.states.pos[:, 0, :]
            setpoint, setpoint_yaw = setpoints(pos, rot, action)
            zeros = jp.zeros((n_envs, 3))
            quat = R.from_euler("z", setpoint_yaw[:, None]).as_quat()
            cmd = jp.concatenate([setpoint, zeros, zeros, quat, zeros], axis=-1)
            return state_control(data, cmd[:, None, :])

        def _terminated(data: SimData, goal: Array, settled: Array) -> tuple[Array, Array, Array]:
            pos = data.states.pos[:, 0, :]
            out_of_bounds = jp.any((pos[:, :2] < lower) | (pos[:, :2] > upper), axis=-1)
            out_of_bounds = out_of_bounds | (pos[:, 2] < floor) | (pos[:, 2] > ceiling)
            at_goal = jp.linalg.norm(pos - goal, axis=-1) < goal_radius
            at_rest = jp.linalg.norm(data.states.vel[:, 0, :], axis=-1) < rest_speed
            # Leaving the goal or picking up speed drops the count, so only an unbroken hold counts
            settled = jp.where(at_goal & at_rest, settled + 1, 0)
            success = settled >= settle_steps
            return out_of_bounds | success, success, settled

        def _reset(
            env: VisionReachEnv, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[VisionReachEnv, tuple[dict[str, Array], dict]]:
            data = env.data
            if seed is not None:
                rng_key = jax.device_put(jax.random.key(seed), jax_device)
                data = data.replace(core=data.core.replace(rng_key=rng_key))
            keys = jax.random.split(data.core.rng_key, 5)
            rng_key, domain_key, imu_key, goal_key, cam_key = keys
            data = data.replace(core=data.core.replace(rng_key=rng_key))
            domain = dr.sample(domain_key, n_envs, ranges)
            goal = jax.random.uniform(goal_key, (n_envs, 3), minval=goal_low, maxval=goal_high)
            data = sim_reset(data, default_data, None)
            frame = _frame(data, domain, cam_key)
            gyro, accel = _imu(data, domain.imu_bias, imu_key)
            env = env.replace(
                data=data,
                domain=domain,
                goal_pos=goal,
                settled=jp.zeros_like(env.settled),
                _marked_for_reset=jp.zeros_like(env._marked_for_reset),
            )
            info = {"success": jp.zeros((n_envs,), dtype=bool)}
            obs = _obs(frame, gyro, accel, goal)
            return env, (obs, info)

        def _step(
            env: VisionReachEnv, action: Array
        ) -> tuple[VisionReachEnv, tuple[dict[str, Array], Array, Array, Array, dict]]:
            data = env.data
            keys = jax.random.split(data.core.rng_key, 7)
            rng_key, gust_key, domain_key, imu_key, drift_key, goal_key, cam_key = keys
            data = data.replace(core=data.core.replace(rng_key=rng_key))
            # 1. apply the action and the disturbance force acting over this step
            data = _apply_action(data, action)
            data = dr.disturb(data, env.domain.wind, gust_key, ranges)
            # 2. step the sim
            data = sim_step(data, n_substeps)
            # 3. terminate on the pre-reset state
            terminated, success, settled = _terminated(data, env.goal_pos, env.settled)
            truncated = data.core.steps[..., 0] / data.core.freq >= max_episode_time
            done = terminated | truncated
            # 4. reward from the pre-reset state. Rendering it as well would double the cost of a
            # step, and nothing consumes an observation from before the reset.
            reward = jp.zeros_like(done, dtype=jp.float32)
            drifted = dr.drift_imu(env.domain.imu_bias, drift_key, ranges)
            final_domain = env.domain._replace(imu_bias=drifted)
            # 5. autoreset done worlds in the same step, redrawing their goal and domain
            domain = dr.replace_done(final_domain, dr.sample(domain_key, n_envs, ranges), done)
            new_goal = jax.random.uniform(goal_key, (n_envs, 3), minval=goal_low, maxval=goal_high)
            goal = jp.where(done[:, None], new_goal, env.goal_pos)
            data = sim_reset(data, default_data, done)
            frame = _frame(data, domain, cam_key)
            gyro, accel = _imu(data, domain.imu_bias, imu_key)
            env = env.replace(
                data=data,
                domain=domain,
                goal_pos=goal,
                settled=jp.where(done, 0, settled),
                steps=data.core.steps // n_substeps,
                _marked_for_reset=jp.zeros_like(done),
            )
            ep_steps = (data.core.steps[:, 0] // n_substeps).astype(jp.int32)
            obs = _obs(frame, gyro, accel, goal)
            info = {"success": success, "episode_steps": ep_steps}
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            sim=sim,
            draw_scene=_draw_scene,
            num_envs=n_envs,
            max_episode_time=max_episode_time,
            control=control,
            dynamics=sim.dynamics,
            device="gpu",
            drone=cls.DRONE,
            freq=cls.FREQ,
            single_action_space=single_action_space,
            action_space=action_space,
            single_observation_space=single_observation_space,
            observation_space=observation_space,
            n_substeps=n_substeps,
            autoreset_mode=AutoresetMode.SAME_STEP,
            goal_pos=jp.zeros((n_envs, 3), device=jax_device),
            domain=dr.sample(jax.random.key(0), n_envs, ranges),
            data=sim.data,
            steps=jp.zeros((n_envs, 1), dtype=jp.int32, device=jax_device),
            settled=jp.zeros((n_envs,), dtype=jp.int32, device=jax_device),
            _marked_for_reset=jp.zeros((n_envs,), dtype=jp.bool_, device=jax_device),
            reset=jax.jit(_reset),
            step=jax.jit(_step),
        )
