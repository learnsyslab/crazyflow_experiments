"""Trajectories the drone follows: periodic curves and sequences of them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

if TYPE_CHECKING:
    from jax import Array

SOFT_START_TIME = 1.0  # s the trajectory ramps in from rest at its start pose


class Trajectory(ABC):
    """Curve the drone follows over time.

    Time zero is where the drone joins the curve. In the `SOFT_START_TIME` before, the curve ramps
    in smoothly from rest for a soft start, earlier it rests at the start of the ramp, and after
    `duration` its end is held at rest, so a controller tracking it hovers outside that window.
    """

    duration: float

    @abstractmethod
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""

    @abstractmethod
    def vel(self, t: Array) -> Array:
        """Return the velocity at time t, shape (..., 3)."""

    @abstractmethod
    def acc(self, t: Array) -> Array:
        """Return the acceleration at time t, shape (..., 3)."""

    @abstractmethod
    def yaw(self, t: Array) -> Array:
        """Return the heading at time t."""

    def warm_up(self, t: float = 0.0):
        """Evaluate every coordinate once, so JAX compiles them before the flight starts."""
        for coordinate in (self.pos, self.vel, self.acc, self.yaw):
            coordinate(jnp.asarray(t))


@dataclass(frozen=True)
class PeriodicTrajectory(Trajectory):
    """Periodic curve around an origin, scaled and permuted onto the world axes."""

    scaling: Array
    cycle_time: float = field(metadata={"static": True})
    duration: float = field(metadata={"static": True})
    origin: Array = field(default_factory=lambda: jnp.zeros(3))
    axes: str = field(default="xyz", metadata={"static": True})
    yaw_mode: str = field(default="constant", metadata={"static": True})
    yaw_setting: float = field(default=0.0, metadata={"static": True})

    @abstractmethod
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""

    @abstractmethod
    def vel(self, t: Array) -> Array:
        """Return the velocity at time t, shape (..., 3)."""

    @abstractmethod
    def acc(self, t: Array) -> Array:
        """Return the acceleration at time t, shape (..., 3)."""

    @property
    def order(self) -> tuple[int, ...]:
        """Return the permutation taking the trajectory's own axes to world axes."""
        return tuple("xyz".index(axis) for axis in self.axes)

    @property
    def extent(self) -> Array:
        """Return the scaling expressed in the trajectory's own axes."""
        return self.scaling[jnp.array([self.order.index(axis) for axis in range(3)])]

    def place(self, x: Array, y: Array, z: Array) -> Array:
        """Return the world position, permuted onto the world axes and offset from the origin."""
        return jnp.stack((x, y, z), axis=-1)[..., jnp.array(self.order)] + self.origin

    def yaw(self, t: Array) -> Array:
        """Return the heading at time t.

        The yaw mode is "constant" at `yaw_setting`, "velocity" for `yaw_setting` turns per cycle,
        "direction" to face along the velocity, or "chirp" for a frequency sweep of amplitude
        `yaw_setting` like `Chirp1D`, which excites the yaw axis while hovering.
        """
        t = jnp.clip(t, -SOFT_START_TIME, self.duration)
        match self.yaw_mode:
            case "constant":
                return jnp.full_like(t, self.yaw_setting)
            case "velocity":
                return 2 * jnp.pi * t / self.cycle_time * self.yaw_setting
            case "direction":
                velocity = self.vel(t)
                return jnp.arctan2(velocity[..., 1], velocity[..., 0])
            case "chirp":
                swept = self.theta(t) / (2 * jnp.pi) * self.cycle_time
                return self.yaw_setting * _linear_chirp(swept, 1e-4, 2.0, self.duration, -90)
            case _:
                raise ValueError(f"Unknown yaw mode {self.yaw_mode}")

    def theta(self, t: Array) -> Array:
        """Return the progress along the trajectory at time t.

        Ramps up smoothly from rest over `SOFT_START_TIME` before t=0 and is held constant after the
        trajectory ends.
        """
        freq = 2.0 * jnp.pi / self.cycle_time
        t = jnp.clip(t, -SOFT_START_TIME, self.duration)
        ramp = (
            freq / (2 * SOFT_START_TIME) * (t + SOFT_START_TIME) ** 2 - freq * SOFT_START_TIME / 2
        )
        return jnp.where(t >= 0, freq * t, ramp)


def spline_derivative(fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Return the vectorized derivative of `fn`."""
    return jax.jit(jnp.vectorize(jax.jacfwd(fn, argnums=1), signature="()->(3)", excluded={0}))


@register_dataclass
@dataclass(frozen=True)
class Ellipse(PeriodicTrajectory):
    """Planar ellipse."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        x = self.extent[0] / 2 * jnp.sin(theta)
        y = self.extent[1] / 2 * jnp.cos(theta)
        return self.place(x, y, jnp.zeros_like(theta))

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class Ellipse3D(PeriodicTrajectory):
    """Ellipse rising and falling twice per cycle."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        x = self.extent[0] / 2 * jnp.sin(theta)
        y = self.extent[1] / 2 * jnp.cos(theta)
        z = self.extent[2] * jnp.cos(theta) * jnp.sin(theta)
        return self.place(x, y, z)

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class Lissajous(PeriodicTrajectory):
    """Planar Lissajous figure with a six to five frequency ratio."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        x = self.extent[0] / 2 * jnp.cos(6 * theta + jnp.pi / 2)
        y = self.extent[1] / 2 * jnp.cos(5 * theta)
        return self.place(x, y, jnp.zeros_like(theta))

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class Figure8(PeriodicTrajectory):
    """Figure of eight that also rises and falls."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        x = self.extent[0] / 2 * jnp.sin(theta)
        y = self.extent[1] / 2 * jnp.sin(2 * theta)
        z = self.extent[2] * jnp.cos(theta) ** 2
        return self.place(x, y, z)

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class Figure8Bent(PeriodicTrajectory):
    """Figure of eight folded out of its plane."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        x = self.extent[0] * (jnp.sin(theta) ** 2 - 0.5)
        y = self.extent[1] / 2 * jnp.cos(theta)
        z = self.extent[2] * jnp.cos(theta) * jnp.sin(theta)
        return self.place(x, y, z)

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class Flower1(PeriodicTrajectory):
    """Flower-like trajectory that also rises and falls."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        rho = 1 - 0.5 * jnp.cos(4 * theta)
        x = self.extent[0] / 2 / jnp.sqrt(2) * rho * jnp.sin(theta)
        y = self.extent[1] / 2 / jnp.sqrt(2) * rho * jnp.cos(theta)
        z = self.extent[2] * jnp.cos(theta) * jnp.sin(theta)
        return self.place(x, y, z)

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class Flower2(PeriodicTrajectory):
    """Flower-like trajectory that also rises and falls."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        x = self.extent[0] / 2 * jnp.sin(3 * theta) * jnp.cos(theta)
        y = self.extent[1] / 2 * jnp.sin(3 * theta) * jnp.sin(theta)
        z = self.extent[2] / 2 * jnp.cos(3 * theta)
        return self.place(x, y, z)

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class Pringle(PeriodicTrajectory):
    """Saddle-shaped closed curve."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        x = self.extent[0] / 2 * jnp.sin(theta) ** 3
        y = self.extent[1] / 2 * jnp.cos(theta) ** 3
        z = self.extent[2] * jnp.cos(theta) * jnp.sin(theta)
        return self.place(x, y, z)

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class Melon(PeriodicTrajectory):
    """Ellipse whose plane rotates by one full turn over all cycles."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        angle = jnp.floor(t / self.cycle_time) * self.cycle_time / self.duration * 2 * jnp.pi
        x = self.extent[0] / 2 * jnp.cos(theta)
        y = self.extent[1] / 2 * jnp.sin(theta) * jnp.sin(angle)
        z = self.extent[2] / 2 * jnp.sin(theta) * jnp.cos(angle)
        return self.place(x, y, z)

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class Spiral(PeriodicTrajectory):
    """Spiralling out and then back in."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        radius = theta / (2 * jnp.pi) * self.cycle_time / self.duration
        radius = jnp.where(radius <= 0.5, radius * 2, (1 - radius) * 2)
        x = self.extent[0] * radius / 2 * jnp.sin(theta)
        y = self.extent[1] * radius / 2 * jnp.cos(theta)
        return self.place(x, y, jnp.zeros_like(theta))

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class Spiral3D(PeriodicTrajectory):
    """Spiral climbing steadily over the run."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        height = theta / (2 * jnp.pi) * self.cycle_time / self.duration - 0.5
        x = self.extent[0] / 2 * jnp.sin(theta)
        y = self.extent[1] / 2 * jnp.cos(theta)
        return self.place(x, y, self.extent[2] * height)

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


def _linear_chirp(t: Array, f_start: float, f_end: float, duration: float, phase: float) -> Array:
    """Return a cosine sweeping linearly from one frequency to another over the duration."""
    sweep = f_start * t + 0.5 * (f_end - f_start) / duration * t**2
    return jnp.cos(2 * jnp.pi * sweep + jnp.deg2rad(phase))


@register_dataclass
@dataclass(frozen=True)
class Chirp1D(PeriodicTrajectory):
    """Frequency sweep along a single axis."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        swept = theta / (2 * jnp.pi) * self.cycle_time
        x = self.extent[0] / 2 * _linear_chirp(swept, 1e-4, 2.0, self.duration, -90)
        return self.place(x, jnp.zeros_like(theta), jnp.zeros_like(theta))

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class Chirp2D(PeriodicTrajectory):
    """Frequency sweep in the plane, the two axes a quarter period apart."""

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        theta = self.theta(t)
        swept = theta / (2 * jnp.pi) * self.duration
        x = self.extent[0] / 2 * _linear_chirp(swept, 1e-6, 1e-1, self.duration, -90)
        y = self.extent[1] / 2 * _linear_chirp(swept, 1e-6, 1e-1, self.duration, 0)
        return self.place(x, y, jnp.zeros_like(theta))

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)


@register_dataclass
@dataclass(frozen=True)
class CompositeTrajectory(Trajectory):
    """Trajectories flown one after another.

    Every trajectory plays for its duration once the previous one has ended. Where one ends at the
    start of the next, the next continues right away. Where they lie apart, a transition of
    `transition_time` seconds interpolates linearly from the end of one to the start of the next.
    The first trajectory keeps its soft start before time zero, the last one holds its end.
    """

    trajectories: tuple[Trajectory, ...]
    transition_time: float = field(default=4.0, metadata={"static": True})
    # Duration of the transition after every trajectory but the last, zero where the trajectories
    # connect. Derived from the trajectories on construction. JAX rebuilds the instance from its
    # flattened form inside jitted functions and passes the value back in, so it never derives from
    # traced values.
    transitions: tuple[float, ...] | None = field(default=None, metadata={"static": True})

    def __post_init__(self):
        """Decide where transitions are needed."""
        if self.transitions is not None:
            return
        transitions = []
        for before, after in zip(self.trajectories[:-1], self.trajectories[1:], strict=True):
            apart = not jnp.allclose(before.pos(before.duration), after.pos(0.0), atol=1e-3)
            transitions.append(self.transition_time if apart else 0.0)
        object.__setattr__(self, "transitions", tuple(transitions))

    @property
    def starts(self) -> tuple[float, ...]:
        """Return the time at which every trajectory reaches its own time zero."""
        starts, t = [0.0], 0.0
        for trajectory, transition in zip(self.trajectories[:-1], self.transitions, strict=True):
            t += trajectory.duration + transition
            starts.append(t)
        return tuple(starts)

    @property
    def duration(self) -> float:
        """Return the time at which the last trajectory ends."""
        return self.starts[-1] + self.trajectories[-1].duration

    @jax.jit
    @partial(jnp.vectorize, signature="()->(3)", excluded={0})
    def pos(self, t: Array) -> Array:
        """Return the position at time t, shape (..., 3)."""
        return self._piecewise(t, lambda trajectory, t: trajectory.pos(t))

    vel = spline_derivative(pos)
    acc = spline_derivative(vel)

    @jax.jit
    def yaw(self, t: Array) -> Array:
        """Return the heading at time t, interpolated linearly through the transitions."""
        return self._piecewise(t, lambda trajectory, t: trajectory.yaw(t))

    def _piecewise(self, t: Array, evaluate: Callable[[Trajectory, Array], Array]) -> Array:
        """Evaluate the piece active at time t, a trajectory at its local time or a transition.

        Args:
            t: Time.
            evaluate: Maps a trajectory and its local time to the coordinate.
        """
        conditions, values = [], []
        for i, (trajectory, start) in enumerate(zip(self.trajectories, self.starts, strict=True)):
            end = start + trajectory.duration
            conditions.append(t < end)
            values.append(evaluate(trajectory, t - start))
            if i < len(self.transitions) and self.transitions[i] > 0:
                tau = self.transitions[i]
                conditions.append(t < end + tau)
                before = evaluate(trajectory, trajectory.duration)
                after = evaluate(self.trajectories[i + 1], 0.0)
                values.append(before + (after - before) * (t - end) / tau)
        return jnp.select(conditions, values, values[-1])  # The last trajectory holds its end


TRAJECTORIES: dict[str, type[PeriodicTrajectory]] = {
    "chirp_1d": Chirp1D,
    "chirp_2d": Chirp2D,
    "ellipse": Ellipse,
    "ellipse_3d": Ellipse3D,
    "figure8": Figure8,
    "figure8_bent": Figure8Bent,
    "flower1": Flower1,
    "flower2": Flower2,
    "lissajous": Lissajous,
    "melon": Melon,
    "pringle": Pringle,
    "spiral": Spiral,
    "spiral_3d": Spiral3D,
}
