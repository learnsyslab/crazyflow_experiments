"""Check the actions of the trajectory controllers."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from crazyflow_experiments.control.config import drone_params  # noqa: E402
from crazyflow_experiments.control.mellinger_controller import MellingerController  # noqa: E402
from crazyflow_experiments.control.state_controller import StateController  # noqa: E402
from crazyflow_experiments.control.trajectories import Ellipse  # noqa: E402

TRAJECTORY = Ellipse(jnp.array([2.0, 2.0, 0.0]), 5.0, 10.0, jnp.array([0.0, 0.0, 1.0]))
START = np.asarray(TRAJECTORY.pos(0.0))
MODEL = "cf2x_L250"
G = 9.81


def hover(t: float, pos: np.ndarray) -> dict:
    """Observation of a drone hovering level at pos."""
    return {
        "pos": np.asarray(pos, dtype=np.float64),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "rpy": np.zeros(3),
        "t": t,
    }


def test_state_controller_plays_the_trajectory():
    """The state command is the reference state of the trajectory at the elapsed time."""
    controller = StateController(hover(10.0, START), TRAJECTORY)
    action = controller(hover(12.5, START))
    assert action.shape == (16,) and action.dtype == np.float64
    assert np.allclose(action[:3], TRAJECTORY.pos(2.5))
    assert np.allclose(action[3:6], TRAJECTORY.vel(2.5))
    assert np.allclose(action[6:9], TRAJECTORY.acc(2.5))
    yaw = R.from_quat(action[9:13]).as_euler("xyz")[2]
    assert yaw == pytest.approx(float(TRAJECTORY.yaw(2.5)))
    assert np.all(action[13:] == 0.0)
    assert not controller.done(hover(19.9, START))
    assert controller.done(hover(20.0, START))


def test_mellinger_controller_commands_attitude_and_thrust():
    """The action is [roll, pitch, yaw, thrust] within the controller's limits."""
    mass = drone_params(MODEL)["mass"]
    controller = MellingerController(hover(0.0, START), TRAJECTORY, MODEL)
    action = controller(hover(0.0, START))
    assert action.shape == (4,)
    assert 0.3 * mass * G <= action[3] <= 1.8 * mass * G
    # Far below the reference the thrust saturates, far to the side the tilt does
    assert controller(hover(0.0, START - [0, 0, 5]))[3] == pytest.approx(1.8 * mass * G)
    limited = MellingerController(hover(0.0, np.zeros(3)), TRAJECTORY, MODEL, tilt_max=0.1)
    action = limited(hover(0.0, START + [5.0, 0.0, 0.0]))
    assert np.abs(action[:2]).max() == pytest.approx(0.1)
    assert not controller.done(hover(9.9, START))
    assert controller.done(hover(10.0, START))


def test_mellinger_controller_heading_variants():
    """The default keeps the heading under tilt, the legacy one lets the yaw turn with it."""
    yawed = Ellipse(
        jnp.array([2.0, 2.0, 0.0]), 5.0, 10.0, jnp.array([0.0, 0.0, 1.0]), yaw_setting=0.5
    )
    tilted = hover(0.0, np.asarray(yawed.pos(0.0)) + [3.0, 0.0, 0.0])
    fixed = MellingerController(hover(0.0, START), yawed, MODEL)(tilted)
    legacy = MellingerController(hover(0.0, START), yawed, MODEL, legacy_heading=True)(tilted)
    assert fixed[2] == pytest.approx(0.5)
    assert legacy[2] != pytest.approx(0.5, abs=1e-3)
    # Hovering on a resting reference, the thrust points up and both variants agree
    resting = Ellipse(jnp.zeros(3), 5.0, 10.0, jnp.array([0.0, 0.0, 1.0]), yaw_setting=0.5)
    level = hover(0.0, np.asarray(resting.pos(0.0)))
    fixed = MellingerController(hover(0.0, START), resting, MODEL)(level)
    legacy = MellingerController(hover(0.0, START), resting, MODEL, legacy_heading=True)(level)
    assert np.allclose(fixed, legacy, atol=1e-6)
    assert fixed[2] == pytest.approx(0.5)


def test_mellinger_controller_mass_from_drone_config():
    """The mass comes from sysid/config_drones.toml, also for vehicles crazyflow does not know."""
    controller = MellingerController(hover(0.0, START), TRAJECTORY, "hb_x500")
    hover_thrust = controller(hover(0.0, START))[3]  # on the planar reference, only gravity in z
    assert hover_thrust == pytest.approx(drone_params("hb_x500")["mass"] * G, rel=1e-3)
