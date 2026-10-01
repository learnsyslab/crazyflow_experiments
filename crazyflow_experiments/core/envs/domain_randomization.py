"""Per-episode randomization of the drone, its disturbances, and its sensors.

The vision section holds the draws the vision env threads through its own keys. The reset and step
sections provide the building blocks for domain randomization, following crazyflow's plugin
examples: every ``build_*`` function returns a stage that plugs into a sim's reset or step pipeline,
so envs compose exactly the randomizations they need. All draws scale the shared default parameters,
which broadcast to one value per world, so repeated resets do not compound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple

import flax.struct as struct
import jax
import jax.numpy as jp
from crazyflow.utils import CORE_NDIM_KEY, leaf_replace
from jax import Array
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData

ResetFn = Callable[["SimData", "SimData", Array | None], "SimData"]
"""Reset pipeline stage: maps (data, default_data, mask) to the modified data."""
StepFn = Callable[["SimData"], "SimData"]
"""Step pipeline stage: maps the sim data to the modified data."""


# region Vision
class Domain(NamedTuple):
    """The draws that stay fixed for one episode."""

    imu_bias: Array  # (worlds, 6), gyro then accelerometer
    wind: Array  # (worlds, 3), steady force in N
    cam_offset: Array  # (worlds, 3), camera mounting offset in m
    cam_rot: Array  # (worlds, 4), camera mounting rotation as xyzw
    cam_fovy: Array  # (worlds,), deviation from the nominal field of view in degrees
    photometry: Array  # (worlds, 5), brightness, contrast, gamma, vignette, exposure


class Ranges(NamedTuple):
    """Spread of every randomized quantity. Zeros switch a quantity off."""

    gyro_bias: float = 0.02  # rad/s
    gyro_noise: tuple[float, float, float] = (0.2, 0.08, 0.04)  # Gyro vibration per axis
    accel_bias: float = 0.2  # m/s²
    accel_noise: tuple[float, float, float] = (0.1, 0.18, 0.18)  # Vibration per axis
    gyro_drift: float = 0.002  # rad/s per step
    accel_drift: float = 0.02  # m/s² per step
    wind_max: float = 0.015  # N
    gust_std: float = 0.005  # N per step
    torque_std: float = 1e-5  # Nm per step
    cam_offset_std: float = 0.005  # m
    cam_angle_std: float = 0.04  # rad
    cam_fovy_range: float = 4.0  # deg
    brightness_std: float = 0.15
    contrast_std: float = 0.2
    gamma_std: float = 0.25
    vignette_max: float = 0.4
    exposure_min: float = 1e-3  # s
    exposure_max: float = 1 / 60  # s, bounded by the frame period
    mass_range: float = 0.05  # Fraction to account for battery load variations
    inertia_range: float = 0.2  # Fraction per axis


def sample(key: Array, worlds: int, ranges: Ranges) -> Domain:
    """Draw a fresh domain for every world.

    Args:
        key: Random key.
        worlds: Number of parallel worlds.
        ranges: Spread of each quantity.

    Returns:
        One Domain with a leading axis over worlds.
    """
    bias_key, wind_key, offset_key, angle_key, fovy_key, photo_key = jax.random.split(key, 6)
    bright_key, contrast_key, gamma_key, vignette_key, exposure_key = jax.random.split(photo_key, 5)
    scale = jp.concatenate([jp.full((3,), ranges.gyro_bias), jp.full((3,), ranges.accel_bias)])
    angles = ranges.cam_angle_std * jax.random.normal(angle_key, (worlds, 3))
    return Domain(
        imu_bias=scale * jax.random.normal(bias_key, (worlds, 6)),
        wind=ranges.wind_max * jax.random.uniform(wind_key, (worlds, 3), minval=-1.0, maxval=1.0),
        cam_offset=ranges.cam_offset_std * jax.random.normal(offset_key, (worlds, 3)),
        cam_rot=R.from_euler("xyz", angles).as_quat(),
        cam_fovy=ranges.cam_fovy_range
        * jax.random.uniform(fovy_key, (worlds,), minval=-1.0, maxval=1.0),
        photometry=jp.stack(
            [
                ranges.brightness_std * jax.random.normal(bright_key, (worlds,)),
                1.0 + ranges.contrast_std * jax.random.normal(contrast_key, (worlds,)),
                jp.exp(ranges.gamma_std * jax.random.normal(gamma_key, (worlds,))),
                ranges.vignette_max * jax.random.uniform(vignette_key, (worlds,)),
                # Auto exposure tracks the lighting, so draw it log uniformly across the range
                jp.exp(
                    jax.random.uniform(
                        exposure_key,
                        (worlds,),
                        minval=jp.log(ranges.exposure_min),
                        maxval=jp.log(max(ranges.exposure_max, ranges.exposure_min)),
                    )
                ),
            ],
            axis=-1,
        ),
    )


def replace_done(current: Domain, fresh: Domain, done: Array) -> Domain:
    """Take the fresh draw only in the worlds whose episode just ended."""

    def pick(new: Array, old: Array) -> Array:
        mask = done.reshape(done.shape + (1,) * (new.ndim - done.ndim))
        return jp.where(mask, new, old)

    return jax.tree.map(pick, fresh, current)


def drift_imu(imu_bias: Array, key: Array, ranges: Ranges) -> Array:
    """Advance the inertial bias by one random walk step, so it moves within an episode."""
    scale = jp.concatenate([jp.full((3,), ranges.gyro_drift), jp.full((3,), ranges.accel_drift)])
    return imu_bias + scale * jax.random.normal(key, imu_bias.shape)


def disturb(data: SimData, wind: Array, key: Array, ranges: Ranges) -> SimData:
    """Apply the steady wind with gusts and a small random torque to the drone."""
    gust_key, torque_key = jax.random.split(key)
    force = wind + ranges.gust_std * jax.random.normal(gust_key, wind.shape)
    torque = ranges.torque_std * jax.random.normal(torque_key, wind.shape)
    states = data.states.replace(force=force[:, None, :], torque=torque[:, None, :])
    return data.replace(states=states)


def randomize_mass_inertia(
    data: SimData, default_data: SimData, mask: Array, key: Array, ranges: Ranges
) -> SimData:
    """Randomize the mass and inertia of the drone."""
    mass_key, inertia_key = jax.random.split(key)
    shape = (data.core.n_worlds, data.core.n_drones)
    mass = default_data.params.mass * jax.random.uniform(
        mass_key, (*shape, 1), minval=1.0 - ranges.mass_range, maxval=1.0 + ranges.mass_range
    )
    # Every principal axis draws its own factor
    minval, maxval = 1.0 - ranges.inertia_range, 1.0 + ranges.inertia_range
    scale = jp.sqrt(jax.random.uniform(inertia_key, (*shape, 3), minval=minval, maxval=maxval))
    J = default_data.params.J * scale[..., :, None] * scale[..., None, :]
    params = leaf_replace(data.params, mask, mass=mass, J=J, J_inv=jp.linalg.inv(J))
    return data.replace(params=params)


# region Reset
class DynamicsRanges(NamedTuple):
    """Spread of every randomized quantity of the drone dynamics. Zeros switch a quantity off.

    Every field feeds the ``build_*`` stage of the same quantity. Fractions scale the nominal
    `ActionDelay` wrapper reads because the lag sits between the policy and the sim rather than in
    the dynamics. Fractions scale the nominal parameter by a factor drawn uniformly from
    [1 - r, 1 + r] at every reset. The wrench is drawn uniformly from [-w, w] per axis at every
    reset and stays fixed for the episode, the noise is white Gaussian noise added to it at every
    sim step.
    """

    mass: float = 0.2  # Fraction
    inertia: float = 0.2  # Fraction, drawn per principal axis
    rotor_dynamics: float = 0.2  # Fraction, drawn per motor and coefficient
    thrust_curve: float = 0.1  # Fraction, drawn per motor and coefficient
    torque_curve: float = 0.1  # Fraction, drawn per motor and coefficient
    force: float = 0.04  # N per axis, about a tenth of the hover thrust
    torque: float = 2e-4  # Nm per axis, the hover thrust acting half a millimeter off the CoM
    force_noise: float = 0.05  # std in N per sim step (normal distribution)
    torque_noise: float = 1e-3  # std in Nm per sim step (normal distribution)
    action_delay: float = 0.04  # s the drone lags a command by, drawn per episode up to this

    def scale(self, factor: float) -> DynamicsRanges:
        """Scale every spread by the same factor, for ablations on the amount of randomization."""
        return DynamicsRanges(*(factor * value for value in self))


def build_randomize_pos_vel_fn(
    pos_min: float, pos_max: float, vel_min: float, vel_max: float
) -> ResetFn:
    """Build a reset stage that offsets the position and draws the velocity of reset worlds.

    Args:
        pos_min: Lower bound of the position offset in m, added to the current position.
        pos_max: Upper bound of the position offset in m, added to the current position.
        vel_min: Lower bound of the velocity draw in m/s.
        vel_max: Upper bound of the velocity draw in m/s.
    """

    def randomize_pos_vel(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
        key, pos_key, vel_key = jax.random.split(data.core.rng_key, 3)
        shape = (data.core.n_worlds, data.core.n_drones, 3)
        pos = data.states.pos
        pos = jax.random.uniform(pos_key, shape, minval=pos + pos_min, maxval=pos + pos_max)
        vel = jax.random.uniform(vel_key, shape, minval=vel_min, maxval=vel_max)
        states = leaf_replace(data.states, mask, pos=pos, vel=vel)
        return data.replace(states=states, core=data.core.replace(rng_key=key))

    return randomize_pos_vel


def build_reset_rotor_fn(rotor_vel: float) -> ResetFn:
    """Build a reset stage that sets the rotor state of reset worlds, e.g. to hover for takeoff.

    Args:
        rotor_vel: Rotor state to set. RPM for the first principles dynamics, thrust in N for the
            fitted models.
    """

    def reset_rotor(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
        shape = (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
        states = leaf_replace(data.states, mask, rotor_vel=jp.full(shape, rotor_vel))
        return data.replace(states=states)

    return reset_rotor


def build_reset_velocity_fn(vel: Array) -> ResetFn:
    """Build a reset stage that sets the velocity of reset worlds to a reference.

    Args:
        vel: Velocity reference, broadcastable to (n_worlds, n_drones, 3).
    """

    def reset_velocity(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
        return data.replace(states=leaf_replace(data.states, mask, vel=vel))

    return reset_velocity


def build_randomize_mass_fn(spread: float) -> ResetFn:
    """Build a reset stage that draws the mass of reset worlds around the nominal mass.

    Args:
        spread: Fraction of the nominal mass, drawn uniformly from [1 - spread, 1 + spread].
    """

    def randomize_mass(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
        key, mass_key = jax.random.split(data.core.rng_key)
        shape = (data.core.n_worlds, data.core.n_drones, 1)
        scale = jax.random.uniform(mass_key, shape, minval=1.0 - spread, maxval=1.0 + spread)
        params = leaf_replace(data.params, mask, mass=default_data.params.mass * scale)
        return data.replace(params=params, core=data.core.replace(rng_key=key))

    return randomize_mass


def build_randomize_inertia_fn(spread: float) -> ResetFn:
    """Build a reset stage that draws the inertia of reset worlds around the nominal inertia.

    Every principal axis draws its own factor, applied as a congruence transform that keeps the
    matrix symmetric and positive definite and updates its inverse exactly.

    Args:
        spread: Fraction of the nominal inertia per axis, drawn from [1 - spread, 1 + spread].
    """

    def randomize_inertia(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
        key, inertia_key = jax.random.split(data.core.rng_key)
        shape = (data.core.n_worlds, data.core.n_drones, 3)
        factor = jax.random.uniform(inertia_key, shape, minval=1.0 - spread, maxval=1.0 + spread)
        scale, inv_scale = jp.sqrt(factor), 1.0 / jp.sqrt(factor)
        J = default_data.params.J * scale[..., :, None] * scale[..., None, :]
        J_inv = default_data.params.J_inv * inv_scale[..., :, None] * inv_scale[..., None, :]
        params = leaf_replace(data.params, mask, J=J, J_inv=J_inv)
        return data.replace(params=params, core=data.core.replace(rng_key=key))

    return randomize_inertia


def build_randomize_rotor_dynamics_fn(spread: float) -> ResetFn:
    """Build a reset stage that draws the rotor dynamics of reset worlds around the nominal ones.

    Every motor of every world draws its own factor for each of the four coefficients, so the
    motors respond differently from the nominal motor model and from each other.

    Args:
        spread: Fraction of the nominal coefficients, drawn from [1 - spread, 1 + spread].
    """

    def randomize_rotor_dynamics(data: SimData, default_data: SimData, mask: Array) -> SimData:
        key, rotor_key = jax.random.split(data.core.rng_key)
        shape = (data.core.n_worlds, data.core.n_drones, 4, 4)
        scale = jax.random.uniform(rotor_key, shape, minval=1.0 - spread, maxval=1.0 + spread)
        coef = default_data.params.rotor_dyn_coef * scale
        params = leaf_replace(data.params, mask, rotor_dyn_coef=coef)
        return data.replace(params=params, core=data.core.replace(rng_key=key))

    return randomize_rotor_dynamics


def build_randomize_thrust_curve_fn(spread: float) -> ResetFn:
    """Build a reset stage that draws the thrust curve of reset worlds around the nominal curve.

    Every motor of every world draws its own factor for each of the three coefficients, so the
    motors produce different thrusts than the onboard controller expects and than each other.

    Args:
        spread: Fraction of the nominal coefficients, drawn from [1 - spread, 1 + spread].
    """

    def randomize_thrust_curve(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
        key, thrust_key = jax.random.split(data.core.rng_key)
        shape = (data.core.n_worlds, data.core.n_drones, 4, 3)
        scale = jax.random.uniform(thrust_key, shape, minval=1.0 - spread, maxval=1.0 + spread)
        curve = default_data.params.rpm2thrust * scale
        params = leaf_replace(data.params, mask, rpm2thrust=curve)
        return data.replace(params=params, core=data.core.replace(rng_key=key))

    return randomize_thrust_curve


def build_randomize_torque_curve_fn(spread: float) -> ResetFn:
    """Build a reset stage that draws the torque curve of reset worlds around the nominal curve.

    Every motor of every world draws its own factor for each of the three coefficients, so the
    motors produce different drag torques than the nominal motor model and than each other.

    Args:
        spread: Fraction of the nominal coefficients, drawn from [1 - spread, 1 + spread].
    """

    def randomize_torque_curve(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
        key, torque_key = jax.random.split(data.core.rng_key)
        shape = (data.core.n_worlds, data.core.n_drones, 4, 3)
        scale = jax.random.uniform(torque_key, shape, minval=1.0 - spread, maxval=1.0 + spread)
        curve = default_data.params.rpm2torque * scale
        params = leaf_replace(data.params, mask, rpm2torque=curve)
        return data.replace(params=params, core=data.core.replace(rng_key=key))

    return randomize_torque_curve


@struct.dataclass
class Wrench:
    """External force and torque on the drone that stay fixed within an episode."""

    force: Array = struct.field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3), world frame, N
    torque: Array = struct.field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3), world frame, Nm


def add_wrench_plugin(data: SimData) -> SimData:
    """Add the zero wrench that the wrench stages draw, hold and disturb.

    Call this before the sim builds its default data, so that resets can restore the plugin.
    """
    force, torque = jp.zeros_like(data.states.force), jp.zeros_like(data.states.torque)
    return data.replace(plugins={**data.plugins, "wrench": Wrench(force=force, torque=torque)})


def build_randomize_wrench_fn(force: float, torque: float) -> ResetFn:
    """Build a reset stage that draws the constant external wrench of reset worlds.

    The draw goes into the wrench plugin added by `add_wrench` and stays fixed until the next
    reset, modeling steady wind and a center of mass offset.

    Args:
        force: Bound of the uniform force draw per axis in N.
        torque: Bound of the uniform torque draw per axis in Nm.
    """

    def randomize_wrench(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
        key, force_key, torque_key = jax.random.split(data.core.rng_key, 3)
        shape = (data.core.n_worlds, data.core.n_drones, 3)
        force_draw = jax.random.uniform(force_key, shape, minval=-force, maxval=force)
        torque_draw = jax.random.uniform(torque_key, shape, minval=-torque, maxval=torque)
        wrench = leaf_replace(data.plugins["wrench"], mask, force=force_draw, torque=torque_draw)
        states = leaf_replace(data.states, mask, force=force_draw, torque=torque_draw)
        plugins = {**data.plugins, "wrench": wrench}
        return data.replace(states=states, plugins=plugins, core=data.core.replace(rng_key=key))

    return randomize_wrench


# region Delay
@struct.dataclass
class Delay:
    """Attitude commands on their way to the drone, and how long each world holds them back."""

    buffer: Array = struct.field(metadata={CORE_NDIM_KEY: 2})  # (N, M, depth + 1, 4), newest last
    held: Array = struct.field(
        metadata={CORE_NDIM_KEY: 1}
    )  # (N, M, 4), command of the current tick
    steps: Array = struct.field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 1), lag in sim steps


def add_delay_plugin(data: SimData, max_steps: int) -> SimData:
    """Add the command buffer that the delay stages draw, fill and read.

    Call this before the sim builds its default data, so that resets can restore the plugin.

    Args:
        data: Sim data to add the plugin to.
        max_steps: Longest delay in sim steps, which sets the depth of the buffer.
    """
    cmd = data.controls.attitude.staged_cmd
    buffer = jp.zeros((*cmd.shape[:-1], max_steps + 1, cmd.shape[-1]))
    steps = jp.zeros((*cmd.shape[:-1], 1), dtype=jp.int32)
    plugin = Delay(buffer=buffer, held=cmd, steps=steps)
    return data.replace(plugins={**data.plugins, "delay": plugin})


def build_randomize_delay_fn(max_steps: int) -> ResetFn:
    """Build a reset stage that draws how long the drone lags its attitude command.

    The draw resolves single simulation steps rather than control steps, so the delay is set an
    order of magnitude finer than the rate the policy runs at.

    Args:
        max_steps: Longest delay in sim steps. The draw is uniform over [0, max_steps].
    """

    def randomize_delay(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
        key, delay_key = jax.random.split(data.core.rng_key)
        delay: Delay = data.plugins["delay"]
        steps = jax.random.randint(delay_key, delay.steps.shape, 0, max_steps + 1)
        cmd = data.controls.attitude.staged_cmd
        plugin = leaf_replace(
            delay, mask, buffer=jp.zeros_like(delay.buffer), held=cmd, steps=steps
        )
        plugins = {**data.plugins, "delay": plugin}
        return data.replace(plugins=plugins, core=data.core.replace(rng_key=key))

    return randomize_delay


def build_delay_attitude_fn(n_substeps: int) -> StepFn:
    """Build a step stage that holds the attitude command back by the delay of each world.

    Insert it before the attitude controller, so the drone latches a command the policy issued a
    few simulation steps ago. The environment writes the command into the staging buffer once per
    control step and the sim holds it from there, so the stage keeps the command of the current
    tick itself and feeds the buffer at the simulation rate, which is what resolves the delay
    finer than a control step.

    Args:
        n_substeps: Simulation steps per control step, which marks a freshly written command.
    """

    def delay_attitude(data: SimData) -> SimData:
        delay: Delay = data.plugins["delay"]
        fresh = (data.core.steps % n_substeps) == 0  # (N, M), the env wrote a command this step
        held = jp.where(fresh[..., None], data.controls.attitude.staged_cmd, delay.held)
        buffer = jp.concatenate([delay.buffer[..., 1:, :], held[..., None, :]], axis=-2)
        index = buffer.shape[-2] - 1 - delay.steps  # Count back from the newest entry
        staged = jp.take_along_axis(buffer, index[..., None], axis=-2)[..., 0, :]
        attitude = data.controls.attitude.replace(staged_cmd=staged)
        plugins = {**data.plugins, "delay": delay.replace(buffer=buffer, held=held)}
        return data.replace(controls=data.controls.replace(attitude=attitude), plugins=plugins)

    return delay_attitude


# region Step
def build_disturb_wrench_fn(force_noise: float, torque_noise: float) -> StepFn:
    """Build a step stage that applies the constant wrench with fresh white noise on top.

    Insert it into the step pipeline before the integration, so the disturbance acts on every sim
    step.

    Args:
        force_noise: Standard deviation of the force noise per axis in N.
        torque_noise: Standard deviation of the torque noise per axis in Nm.
    """

    def disturb_wrench(data: SimData) -> SimData:
        key, force_key, torque_key = jax.random.split(data.core.rng_key, 3)
        wrench: Wrench = data.plugins["wrench"]
        shape = wrench.force.shape
        force = wrench.force + force_noise * jax.random.normal(force_key, shape)
        torque = wrench.torque + torque_noise * jax.random.normal(torque_key, shape)
        states = data.states.replace(force=force, torque=torque)
        return data.replace(states=states, core=data.core.replace(rng_key=key))

    return disturb_wrench
