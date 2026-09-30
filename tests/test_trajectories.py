"""Check that every trajectory provides the coordinates a controller asks for."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.control.trajectories import (  # noqa: E402
    TRAJECTORIES,
    Chirp1D,
    CompositeTrajectory,
    Ellipse,
)

SCALING = jnp.array([2.0, 1.5, 1.0])
ORIGIN = jnp.array([0.5, -0.25, 1.25])
CYCLE_TIME = 5.0
DURATION = 15.0
INSTANCES = {name: cls(SCALING, CYCLE_TIME, DURATION, ORIGIN) for name, cls in TRAJECTORIES.items()}


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_provides_position_velocity_and_acceleration(name: str):
    """Every entry provides pos, vel and acc with fitting shapes."""
    trajectory = INSTANCES[name]
    t = np.linspace(0.0, DURATION, 51)
    assert trajectory.pos(t).shape == trajectory.vel(t).shape == trajectory.acc(t).shape == (51, 3)


def ellipse(origin: list[float], cycle_time: float = 4.0, yaw_setting: float = 0.0) -> Ellipse:
    """Two cycles of an ellipse around the origin."""
    return Ellipse(SCALING, cycle_time, 2 * cycle_time, jnp.array(origin), yaw_setting=yaw_setting)


def test_composite_plays_the_trajectories_one_after_another():
    """Each trajectory runs at its own local time, bridged by a transition where they lie apart."""
    first, second = ellipse([0.0, 0.0, 1.0]), ellipse([2.0, 0.0, 1.0], cycle_time=2.5)
    composite = CompositeTrajectory((first, second), transition_time=2.0)
    assert composite.transitions == (2.0,)
    assert composite.starts == (0.0, 10.0)  # 8 s first, 2 s transition
    assert composite.duration == 15.0
    t = np.linspace(-1.0, 8.0, 19)  # including the soft start of the first trajectory
    assert np.allclose(composite.pos(t), first.pos(t), atol=1e-5)
    assert np.allclose(composite.yaw(t), first.yaw(t), atol=1e-5)
    t = np.linspace(0.0, 5.0, 11)
    assert np.allclose(composite.pos(10.0 + t), second.pos(t), atol=1e-5)
    assert np.allclose(composite.vel(10.0 + t), second.vel(t), atol=1e-4)
    assert np.allclose(composite.acc(10.0 + t), second.acc(t), atol=1e-3)
    assert np.allclose(composite.pos(composite.duration + 10.0), second.pos(5.0 + 10.0))  # holds


def test_composite_connects_matching_trajectories_directly():
    """No transition is inserted where one trajectory ends at the start of the next."""
    first, second = ellipse([0.0, 0.0, 1.0]), ellipse([0.0, 0.0, 1.0], cycle_time=3.0)
    composite = CompositeTrajectory((first, second))
    assert composite.transitions == (0.0,)
    assert composite.duration == 14.0
    assert np.allclose(composite.pos(8.0), second.pos(0.0), atol=1e-5)


def test_composite_transition_is_linear():
    """The transition runs straight from the end of one trajectory to the start of the next."""
    first, second = ellipse([0.0, 0.0, 1.0]), ellipse([2.0, 0.0, 1.0])
    composite = CompositeTrajectory((first, second), transition_time=2.0)
    p0, p1 = first.pos(8.0), second.pos(0.0)
    assert np.allclose(composite.pos(8.0), p0, atol=1e-5)
    assert np.allclose(composite.pos(9.0), (p0 + p1) / 2, atol=1e-5)
    assert np.allclose(composite.pos(10.0), p1, atol=1e-5)
    t = np.linspace(8.1, 9.9, 10)
    assert np.allclose(composite.vel(t), (p1 - p0) / 2.0, atol=1e-4)
    assert np.allclose(composite.acc(t), 0.0, atol=1e-3)


def test_composite_interpolates_the_yaw_through_the_transition():
    first, second = ellipse([0.0, 0.0, 1.0]), ellipse([2.0, 0.0, 1.0], yaw_setting=1.0)
    composite = CompositeTrajectory((first, second), transition_time=2.0)
    assert np.allclose(composite.yaw(np.array([7.0, 9.0, 11.0])), [0.0, 0.5, 1.0], atol=1e-5)


def test_chirp_yaw_mode_sweeps_the_heading_while_hovering():
    chirp = Chirp1D(jnp.zeros(3), 10.0, 30.0, ORIGIN, yaw_mode="chirp", yaw_setting=0.7)
    t = np.linspace(0.0, 30.0, 3001)
    assert np.allclose(chirp.pos(t), ORIGIN, atol=1e-6)
    yaw = np.asarray(chirp.yaw(t))
    assert abs(yaw[0]) < 1e-5
    assert np.max(np.abs(yaw)) == pytest.approx(0.7, abs=1e-2)
