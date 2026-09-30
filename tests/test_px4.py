"""Drive the PX4 interface and the flight phases of a deployment against a fake MAVROS node."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import threading  # noqa: E402
import time  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rclpy  # noqa: E402
import toml  # noqa: E402
from crazyflow.control import Control  # noqa: E402
from mavros_msgs.msg import PositionTarget  # noqa: E402
from rclpy.signals import SignalHandlerOptions  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from crazyflow_experiments.control.config import (  # noqa: E402
    DRONES_CONFIG,
    drone_params,
)
from crazyflow_experiments.control.controller import Controller  # noqa: E402
from crazyflow_experiments.deploy.experiment import track  # noqa: E402
from crazyflow_experiments.deploy.px4 import Drone, thrust2pwm  # noqa: E402
from tests.conftest import PACKAGE_ROOT  # noqa: E402
from tests.px4_fixtures import FakeMavros  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


THRUST_CURVE = (24.33, 17.72, 0.0)  # N per motor over the normalized command
THRUST_MIN, THRUST_MAX = 0.0, 24.33 + 17.72  # N per motor, the whole command range
PX4_CONFIGS = (PACKAGE_ROOT / "sysid/config.toml", PACKAGE_ROOT / "sim2real/config.toml")
# The fake reports in MAVROS's ENU frame, the interface in the global frame (north, west, up) with
# its origin at home
GLOBAL_FROM_ENU = R.from_euler("z", -np.pi / 2)


def to_global(pos_enu: np.ndarray, home_enu: np.ndarray) -> np.ndarray:
    return GLOBAL_FROM_ENU.apply(pos_enu - home_enu)


def xyz(field: object) -> np.ndarray:
    return np.array([field.x, field.y, field.z])


def test_thrust2pwm():
    """The command is the positive root of the per motor curve, clipped to [0, 1]."""
    assert thrust2pwm(0.0, THRUST_CURVE, THRUST_MIN, THRUST_MAX) == 0.0
    assert thrust2pwm(-10.0, THRUST_CURVE, THRUST_MIN, THRUST_MAX) == 0.0
    per_motor = 24.33 * 0.5**2 + 17.72 * 0.5
    assert thrust2pwm(4 * per_motor, THRUST_CURVE, THRUST_MIN, THRUST_MAX) == pytest.approx(0.5)
    assert thrust2pwm(4 * THRUST_MAX, THRUST_CURVE, THRUST_MIN, THRUST_MAX) == pytest.approx(1.0)
    assert thrust2pwm(500.0, THRUST_CURVE, THRUST_MIN, THRUST_MAX) == 1.0


def test_thrust2pwm_linear_curve():
    """A zero quadratic coefficient, e.g. the proportional fit, inverts without dividing by it."""
    curve = (0.0, 12.3, 0.0)
    assert thrust2pwm(4 * 6.15, curve, 0.0, 12.3) == pytest.approx(0.5)
    assert thrust2pwm(0.0, curve, 0.0, 12.3) == 0.0
    assert thrust2pwm(500.0, curve, 0.0, 12.3) == 1.0
    assert thrust2pwm(4 * 3.0, (0.0, 10.0, 1.0), 0.0, 11.0) == pytest.approx(0.2)


def test_thrust2pwm_limits():
    """The thrust limits clip the command to the commands they map to."""
    thrust_min, thrust_max = np.polyval(THRUST_CURVE, 0.1), np.polyval(THRUST_CURVE, 0.9)
    assert thrust2pwm(0.0, THRUST_CURVE, thrust_min, thrust_max) == pytest.approx(0.1)
    assert thrust2pwm(500.0, THRUST_CURVE, thrust_min, thrust_max) == pytest.approx(0.9)
    per_motor = np.polyval(THRUST_CURVE, 0.5)
    assert thrust2pwm(4 * per_motor, THRUST_CURVE, thrust_min, thrust_max) == pytest.approx(0.5)


def test_thrust2pwm_concave_curve():
    """A curve bending down with an offset stays real and in range when the thrust saturates."""
    curve = (-1.5248, 14.0907, -0.4280)  # measured x500 curve
    thrust_min, thrust_max = 1.0, 12.1
    # Beyond the curve's maximum the raw root is complex, the limits keep the command on the curve
    cmd_max = thrust2pwm(500.0, curve, thrust_min, thrust_max)
    cmd_min = thrust2pwm(-10.0, curve, thrust_min, thrust_max)
    assert 0.0 < cmd_min < cmd_max < 1.0
    assert np.polyval(curve, cmd_max) == pytest.approx(thrust_max)
    assert np.polyval(curve, cmd_min) == pytest.approx(thrust_min)
    per_motor = np.polyval(curve, 0.7)
    assert thrust2pwm(4 * per_motor, curve, thrust_min, thrust_max) == pytest.approx(0.7)


@pytest.mark.parametrize("config_path", PX4_CONFIGS, ids=lambda p: p.parent.name)
def test_configured_model_has_params(config_path: Path):
    """The model a config selects has its table in sysid/config_drones.toml."""
    params = drone_params(toml.load(config_path)["drone"]["model"])
    assert {"mass", "pwm_min", "pwm_max", "thrust_curve", "thrust_min", "thrust_max"} <= set(params)


def test_configured_thrust_curves():
    """Every table of sysid/config_drones.toml holds a quadratic increasing over the command range."""
    drones = toml.load(DRONES_CONFIG)
    for model, params in drones.items():
        curve, thrust_min, thrust_max = (
            params["thrust_curve"],
            params["thrust_min"],
            params["thrust_max"],
        )
        assert len(curve) == 3, f"{model}: thrust = a u^2 + b u + c"
        assert 0 <= params["pwm_min"] < params["pwm_max"], f"{model}: pwm range"
        cmd = np.linspace(0.0, 1.0, 101)
        thrust = np.polyval(curve, cmd)
        assert np.all(np.diff(thrust) > 0), f"{model}: not invertible on [0, 1]"
        assert thrust[0] <= thrust_min < thrust_max <= thrust[-1], f"{model}: limits outside curve"
        inside = (thrust >= thrust_min) & (thrust <= thrust_max)
        assert np.allclose(
            [thrust2pwm(4 * t, curve, thrust_min, thrust_max) for t in thrust[inside]], cmd[inside]
        ), model


@pytest.fixture
def fake() -> Iterator[FakeMavros]:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    fake = FakeMavros()
    yield fake
    fake.close()
    rclpy.shutdown()


@pytest.fixture
def drone(fake: FakeMavros) -> Iterator[Drone]:
    drone = Drone(
        thrust_curve=THRUST_CURVE,
        thrust_min=THRUST_MIN,
        thrust_max=THRUST_MAX,
        ctrl_interface=Control.attitude,
        stream_rate=100.0,
        rtk_outdated_after=0.1,  # shorter than the default, so the watchdog fires within a test
    )
    yield drone
    drone.close()


@pytest.fixture
def mocap_drone(fake: FakeMavros) -> Iterator[Drone]:
    drone = Drone(
        thrust_curve=THRUST_CURVE,
        thrust_min=THRUST_MIN,
        thrust_max=THRUST_MAX,
        ctrl_interface=Control.attitude,
        stream_rate=100.0,
        localization="mocap",
    )
    yield drone
    drone.close()


def switch_to_offboard(fake: FakeMavros, drone: Drone):
    threading.Timer(0.2, fake.set_mode, args=("OFFBOARD",)).start()
    drone.wait_for_offboard()
    assert drone.offboard


def test_stream_rates_are_requested(fake: FakeMavros, drone: Drone):
    """The attitude and local position messages are requested at the stream rate, GPS at its own."""
    assert sorted(fake.interval_requests) == [(24, 20.0), (30, 100.0), (31, 100.0), (32, 100.0)]


def test_observation(fake: FakeMavros, drone: Drone):
    """The observation is relative to home with x north, and carries the IMU body rates."""
    obs = drone.obs()
    assert np.allclose(obs["pos"], to_global(fake.pos, fake.home))
    assert np.allclose(obs["pos"], [3.0, -0.5, 1.0])  # 3 m north, 0.5 m east, 1 m up of home
    assert np.allclose(obs["vel"], GLOBAL_FROM_ENU.apply(fake.vel))
    assert np.allclose(obs["ang_vel"], fake.gyro)
    assert obs["rpy"][2] == pytest.approx(fake.yaw - np.pi / 2)  # ENU yaw counts from east
    assert np.allclose(obs["quat"], R.from_euler("z", fake.yaw - np.pi / 2).as_quat())
    assert not obs["is_outdated"]
    gps = obs["gps"]  # The receiver's solution in SI units, with its own timestamp
    assert gps["fix_type"] == 6 and gps["satellites"] == 22 and gps["dgps_numch"] == 18
    assert gps["lat"] == pytest.approx(43.5) and gps["lon"] == pytest.approx(-79.5)
    assert gps["alt"] == pytest.approx(120.0)
    assert gps["hdop"] == pytest.approx(0.8) and gps["vdop"] == pytest.approx(1.2)
    assert (gps["h_acc"], gps["v_acc"], gps["vel_acc"]) == pytest.approx((0.014, 0.021, 0.03))
    assert gps["dgps_age"] == pytest.approx(1.2)
    assert time.time() - gps["t"] < 1.0
    time.sleep(0.3)
    assert drone.obs()["gps"]["t"] > gps["t"], "a new receiver sample carries a new timestamp"
    assert drone.mode == "AUTO.LOITER" and not drone.offboard and drone.armed


def test_rtk_loss_outdates_the_observation(fake: FakeMavros, drone: Drone):
    """With "gps" the RTK fix is the measurement, so losing it outdates the observation."""
    assert not drone.obs()["is_outdated"]
    fake.gps["fix_type"] = 3  # a plain 3D fix, the receiver still streams at its full rate
    time.sleep(0.2)
    assert drone.obs()["is_outdated"], "the RTK fix is gone, the estimate cannot be trusted"
    assert drone.obs()["gps"]["fix_type"] == 3
    fake.gps["fix_type"] = 6
    time.sleep(0.2)
    assert not drone.obs()["is_outdated"]
    fake.sensors_on = False  # the whole MAVROS stream stops
    time.sleep(0.2)
    assert drone.obs()["is_outdated"]


def test_mocap_watches_the_vision_pose(fake: FakeMavros, mocap_drone: Drone):
    """With "mocap" the global frame sits at the estimator origin and the vision pose is watched."""
    assert np.allclose(mocap_drone.obs()["pos"], GLOBAL_FROM_ENU.apply(fake.pos))
    assert not mocap_drone.obs()["is_outdated"]
    fake.gps["fix_type"] = 0  # no GNSS at all indoors, which must not end the flight
    time.sleep(0.2)
    assert not mocap_drone.obs()["is_outdated"]
    fake.vision_on = False  # the motion capture drops out while PX4 dead reckons on the IMU
    time.sleep(0.3)
    assert mocap_drone.obs()["is_outdated"]


def test_hold_stream_before_and_after_offboard(fake: FakeMavros, drone: Drone):
    """The current pose is streamed until OFFBOARD, then the hold setpoint is frozen."""
    since = time.monotonic()
    time.sleep(0.5)
    targets = fake.targets("position", since)
    assert 15 <= len(targets) <= 35, "setpoints repeat at roughly 50 Hz"
    assert targets[-1].coordinate_frame == PositionTarget.FRAME_LOCAL_NED
    assert targets[-1].type_mask == PositionTarget.IGNORE_YAW_RATE
    assert np.allclose(xyz(targets[-1].position), fake.pos)  # back in ENU
    assert targets[-1].yaw == pytest.approx(fake.yaw)
    assert not fake.targets("attitude")

    switch_to_offboard(fake, drone)
    fake.pos = fake.pos + 0.2  # the vehicle drifts, the hold setpoint must not follow
    since = time.monotonic()
    time.sleep(0.5)
    targets = fake.targets("position", since)
    assert 15 <= len(targets) <= 35
    assert np.allclose(xyz(targets[-1].position), [1.0, 2.0, 3.0])


def test_fly_to(fake: FakeMavros, drone: Drone):
    """The flight to a pose ramps position setpoints from the current pose to the target."""
    switch_to_offboard(fake, drone)
    since = time.monotonic()
    drone.fly_to(np.array([3.5, -0.5, 1.0]), 0.0, speed=1.0, hover_time=0.2)
    targets = fake.targets("position", since)
    assert np.allclose(xyz(targets[0].position), fake.pos, atol=0.05)
    assert np.allclose(xyz(targets[-1].position), [1.0, 2.5, 3.0])  # 0.5 m north of the start
    assert targets[-1].yaw == pytest.approx(np.pi / 2)  # facing north


def test_attitude_setpoint(fake: FakeMavros, drone: Drone):
    """Attitude actions arrive as one quaternion in ENU and normalized thrust, and are not repeated."""
    switch_to_offboard(fake, drone)
    since = time.monotonic()
    action = np.array([0.1, -0.2, 0.3, 40.0])
    drone.send_action(action)
    time.sleep(0.4)
    targets = fake.targets("attitude", since)
    assert len(targets) == 1
    q = targets[0].orientation
    expected = GLOBAL_FROM_ENU.inv() * R.from_euler("xyz", action[:3])
    assert np.allclose([q.x, q.y, q.z, q.w], expected.as_quat())
    assert targets[0].thrust == pytest.approx(
        thrust2pwm(40.0, THRUST_CURVE, THRUST_MIN, THRUST_MAX)
    )
    assert targets[0].type_mask == 7  # body rates ignored
    assert not fake.targets("position", since)


def test_state_setpoint_hold_and_takeover(fake: FakeMavros, drone: Drone):
    """State actions carry the feedforward, hold brakes ahead and repeats, takeover releases."""
    switch_to_offboard(fake, drone)
    drone.ctrl_interface = Control.state
    since = time.monotonic()
    pos, vel, acc, yaw = [1, 2, 3], [0.1, 0.2, 0.3], [0.01, 0.02, 0.03], 0.7
    drone.send_action(
        np.concatenate((pos, vel, acc, R.from_euler("z", yaw).as_quat(), np.zeros(3)))
    )
    expected_pos = GLOBAL_FROM_ENU.inv().apply(pos) + fake.home
    deadline = time.monotonic() + 0.5
    targets = []
    while time.monotonic() < deadline:
        targets = [
            target
            for target in fake.targets("position", since)
            if np.allclose(xyz(target.position), expected_pos)
        ]
        if targets:
            break
        time.sleep(0.01)
    assert targets, "the commanded state setpoint did not arrive"
    target = targets[0]
    assert np.allclose(xyz(target.velocity), GLOBAL_FROM_ENU.inv().apply(vel))
    assert np.allclose(xyz(target.acceleration_or_force), GLOBAL_FROM_ENU.inv().apply(acc))
    assert target.yaw == pytest.approx(yaw + np.pi / 2)

    since = time.monotonic()
    drone.hold()
    time.sleep(0.5)
    targets = fake.targets("position", since)
    assert len(targets) >= 15, "the hold setpoint is repeated"
    assert np.allclose(xyz(targets[0].position), fake.pos + fake.vel * 0.5)

    threading.Timer(0.2, fake.set_mode, args=("POSCTL",)).start()
    drone.wait_for_takeover()
    assert not drone.offboard


class ConstantAttitude(Controller):
    """Attitude controller that records the trajectory time of every observation it sees."""

    def __init__(self, steps: int):
        """Take over for `steps` control steps."""
        self.steps = steps
        self.times: list[float] = []
        self._t_start = 0.0  # `track` sets the clock when the controller joins the trajectory

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return a level attitude command and record the trajectory time of the observation."""
        self.times.append(obs["t"] - self._t_start)
        return np.array([0.0, 0.0, 0.0, 40.0])

    def done(self, obs: dict[str, np.ndarray]) -> bool:
        """Stop once the controller has flown its steps."""
        return len(self.times) >= self.steps


def test_track_records_trajectory_time(fake: FakeMavros, drone: Drone):
    """The controller joins the trajectory at the start time, and the recording is timed from there."""
    switch_to_offboard(fake, drone)
    controller = ConstantAttitude(steps=5)
    since = time.monotonic()
    data = track(drone, controller, 50.0, -1.0)
    time.sleep(0.1)  # Let the last setpoint arrive at the fake

    assert len(fake.targets("attitude", since)) == len(data) == 5
    assert controller.times[0] == pytest.approx(-1.0, abs=0.05), (
        "the clock starts at the start time"
    )
    assert data[0]["t"] == controller.times[0], "the recording is in trajectory time"
    assert data[-1]["t"] > data[0]["t"], "the trajectory time runs on"
    assert data[-1]["u"].shape == (4,)
