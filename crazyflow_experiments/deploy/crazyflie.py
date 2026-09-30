"""Interface to a Crazyflie over the radio link."""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import logging
import multiprocessing as mp
import struct
import time
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cflib
import numpy as np
import rclpy
from cflib.crazyflie import Crazyflie, Localization
from cflib.crazyflie.log import LogConfig
from cflib.crtp import init_drivers
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.utils.power_switch import PowerSwitch
from crazyflow.control import Control, load_params
from crazyflow.drones import Drone as Drones
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BOOT_TIME = 4.0  # s, the firmware answers CRTP about 3s after the STM is powered up
CONNECT_ATTEMPTS = 3
CONNECT_TIMEOUT = 15.0  # s, the TOCs take about 8s to download when the cache misses
MIN_RETURN_HEIGHT = 0.5  # m, bounds of the height the drone keeps when returning home
MAX_RETURN_HEIGHT = 2.5  # m


class Drone:
    """Interface to the real Crazyflie."""

    def __init__(
        self,
        drone_id: int,
        channel: int,
        model: str,
        pos_landing: list[float],
        pos_update_freq: float,
        ctrl_interface: Control,
        log_variables: list[str] | None = None,
        log_callback: Callable | None = None,
    ):
        """Connect to the drone, reset its estimator and spin up the rotors.

        Args:
            drone_id: Drone ID.
            channel: Drone radio channel.
            model: Drone model.
            pos_landing: Landing position after the flight.
            pos_update_freq: Position update frequency for the onboard estimator.
            ctrl_interface: Control interface.
            log_variables: Firmware variables to stream back.
            log_callback: Receives the streamed variables.
        """
        assert model in Drones, f"Unknown drone model {model}"
        self.id = drone_id
        self.name = f"cf{drone_id}"
        self.model = model
        self.params = load_params("mellinger", model)["core"]
        self.pos_landing = pos_landing
        self.pos_update_freq = pos_update_freq
        self._last_estimator_update = 0.0
        self.ctrl_interface = ctrl_interface
        self._log_variables = log_variables
        self._log_callback = log_callback

        if not rclpy.ok():
            rclpy.init()
        self._ros_connector = ROSConnector(
            estimator_names=[self.name], cmd_topic=f"/drones/{self.name}/command", timeout=10.0
        )
        self._ros_connector_tf = ROSConnector(tf_names=[self.name], timeout=10.0)

        self.healthy = mp.Event()
        self.healthy.set()
        self.cf = self._connect(0, channel, drone_id)
        time.sleep(0.5)
        self._reset()

    def obs(self) -> dict[str, NDArray]:
        """Generates an observation containing pos, quat, rpy, vel, ang_vel, and is_outdated."""
        obs = {
            "pos": self._ros_connector.pos[self.name],
            "quat": self._ros_connector.quat[self.name],
            "vel": self._ros_connector.vel[self.name],
            "ang_vel": self._ros_connector.ang_vel[self.name],
            "is_outdated": self._ros_connector.is_outdated[self.name],
            "pos_raw": self._ros_connector_tf.pos[self.name],
            "quat_raw": self._ros_connector_tf.quat[self.name],
        }
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        return obs

    def send_state(self, pos: NDArray, vel: NDArray, acc: NDArray, yaw: float):
        """Send a full state setpoint to the drone.

        Args:
            pos: Desired position as (3,).
            vel: Desired velocity as (3,).
            acc: Desired acceleration as (3,).
            yaw: Desired yaw in radians.
        """
        quat = R.from_euler("z", yaw).as_quat()
        self.cf.commander.send_full_state_setpoint(pos, vel, acc, quat, 0.0, 0.0, 0.0)
        self._ros_connector.publish_cmd(np.concatenate((pos, vel, acc, [yaw])))

    def send_action(self, action: NDArray):
        """Send the action to the drone."""
        match self.ctrl_interface:
            case Control.attitude:
                action = (*np.rad2deg(action[:3]), int(self._thrust2pwm(action[3])))
                self.cf.commander.send_setpoint(*action)
                self._ros_connector.publish_cmd(action)
            case Control.state:
                yaw = R.from_quat(action[9:13]).as_euler("xyz")[2]
                self.send_state(action[:3], action[3:6], action[6:9], yaw)
            case Control.rotor_vel:  # Onboard only, nothing should be sent to the drone
                assert np.all(action == 0), "Rotor velocity control must be onboard."
                self.cf.commander.send_setpoint(0, 0, 0, 0)
            case _:
                raise NotImplementedError(f"Unknown control interface {self.ctrl_interface}")

    def update_estimators(self, pos: NDArray | None = None, quat: NDArray | None = None):
        """Send the drone its pose to update the onboard estimators.

        Rate limited to `pos_update_freq`. Sending more often degrades its flight performance.
        """
        if (now := time.perf_counter()) - self._last_estimator_update < 1 / self.pos_update_freq:
            return
        self._last_estimator_update = now
        obs = self.obs()
        pos = pos if pos is not None else obs["pos"]
        quat = quat if quat is not None else obs["quat"]
        self.cf.extpos.send_extpose(*pos, *quat)

    def close(self):
        """Closes the deployment."""
        try:
            self._emergency_stop()
            self.cf.close_link()
        finally:
            # Close all ROS connections
            self._ros_connector.close()
            self._ros_connector_tf.close()

    def _emergency_stop(self):
        """Stop the motors immediately. The drone falls and has to be rebooted to fly again."""
        pk = CRTPPacket()
        pk.port = CRTPPort.LOCALIZATION
        pk.channel = Localization.GENERIC_CH
        pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
        self.cf.send_packet(pk)

    def _thrust2pwm(self, thrust: float) -> float:
        pwm = thrust / self.params["thrust_max"] / 4 * self.params["pwm_max"]
        return np.clip(pwm, self.params["pwm_min"], self.params["pwm_max"])

    def _connect(self, radio_id: int, radio_channel: int, drone_id: int) -> Crazyflie:
        cflib.crtp.init_drivers()
        uri = f"radio://{radio_id}/{radio_channel}/2M/E7E7E7E7" + f"{drone_id:02x}".upper()

        power_switch = PowerSwitch(uri)
        power_switch.stm_power_cycle()
        power_switch.close()
        time.sleep(BOOT_TIME)

        cf = Crazyflie(rw_cache=str(Path(__file__).parent / ".cache"))
        connected_event = mp.Event()

        def cb_logger_data(timestamp: float, data: dict, logconf: LogConfig):
            """Callback froma the log API when data arrives."""
            logger.info(f"[{timestamp / 1000.0:.2f}s] {logconf.name}: {data}")

        def cb_logger_error(logconf: LogConfig, msg: str):
            """Callback from the log API when an error occurs."""
            logger.error("Error when logging %s: %s" % (logconf.name, msg))

        def cb_fully_connected(_):
            """This callback is called when a Crazyflie has been fully connected.

            This includes that the TOCs have been downloaded.
            """
            if self._log_variables is not None:
                # Since the maximum logged bytes per package/logger is 31,
                # we need to split the variables into multiple loggers if there are too many.
                # We assume each variable is 4 bytes, so we can log at most 7 variables per logger.
                max_variables = 6  # 7 doesn't work for some reason, so we use 6 to be safe
                if (total_len := len(self._log_variables)) > max_variables:
                    logger.warning(
                        f"Too many log variables ({total_len}), splitting into multiple loggers"
                    )
                    log_variables_split = [
                        self._log_variables[i : i + max_variables]
                        for i in range(0, len(self._log_variables), max_variables)
                    ]
                else:
                    log_variables_split = [self._log_variables]
                for i, log_variables in enumerate(log_variables_split):
                    cf_logger = LogConfig(name=f"cf_logger_{i}", period_in_ms=10)
                    for var_name in log_variables:
                        cf_logger.add_variable(var_name)
                    cf.log.add_config(cf_logger)
                    callback = self._log_callback if self._log_callback else cb_logger_data
                    cf_logger.data_received_cb.add_callback(callback)
                    cf_logger.error_cb.add_callback(cb_logger_error)
                    cf_logger.start()

            connected_event.set()

        cf.fully_connected.add_callback(cb_fully_connected)
        cf.connection_failed.add_callback(
            lambda _, msg: logger.warning(f"Connection failed: {msg}")
        )
        cf.connection_lost.add_callback(lambda _, msg: logger.warning(f"Connection lost: {msg}"))
        logger.info(f"Waiting for drone {drone_id} to connect...")
        # The connection setup starts with a request that cflib sends without expecting a reply, so
        # it is never retried. A single lost packet, or a firmware that is still booting, stalls the
        # setup for good without ever reporting an error. Reopening the link is the only way out.
        connected = False
        for attempt in range(CONNECT_ATTEMPTS):
            cf.open_link(uri)
            if connected := connected_event.wait(CONNECT_TIMEOUT):
                break
            logger.warning(f"Connection setup of drone {drone_id} stalled, reopening the link")
            cf.close_link()
        assert connected, f"Drone {drone_id} failed to connect to {uri} in {attempt + 1} attempts"
        cf.disconnected.add_callback(lambda _: self.healthy.clear())  # Only a loss of the link now
        logger.info(f"Drone {drone_id} connected to {uri}")

        return cf

    def _reset(self):
        init_drivers()
        self._apply_settings()
        pos = self._ros_connector.pos[self.name]
        # Reset Kalman filter values
        self.cf.param.set_value("kalman.initialX", pos[0])
        self.cf.param.set_value("kalman.initialY", pos[1])
        self.cf.param.set_value("kalman.initialZ", pos[2])
        quat = self._ros_connector.quat[self.name]
        yaw = R.from_quat(quat).as_euler("xyz", degrees=False)[2]
        self.cf.param.set_value("kalman.initialYaw", yaw)
        self.cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self.cf.param.set_value("kalman.resetEstimation", "0")
        self.cf.commander.send_setpoint(0, 0, 0, 0)  # unlock drone
        self.cf.supervisor.send_arming_request(True)  # unlock brushless drone
        self.cf.commander.send_setpoint(0.0, 0.0, 0.0, 7000)  # send min thrust to start motors
        time.sleep(1.5)  # Wait for settings to be applied and propellers to spin up

    def _apply_settings(self):
        """Apply firmware settings to the drone.

        Note:
            These settings are also required to make the high-level drone commander work properly.
        """
        # Estimator setting;  1: complementary, 2: kalman -> Manual test: kalman significantly better!
        self.cf.param.set_value("stabilizer.estimator", 2)
        # enable/disable tumble control. Required 0 for agressive maneuvers
        self.cf.param.set_value("supervisor.tmblChckEn", 0)
        # Choose controller: 1: PID; 2:Mellinger
        self.cf.param.set_value("stabilizer.controller", 2)
        # rate: 0, angle: 1
        self.cf.param.set_value("flightmode.stabModeRoll", 1)
        self.cf.param.set_value("flightmode.stabModePitch", 1)
        self.cf.param.set_value("flightmode.stabModeYaw", 1)
        time.sleep(0.1)  # Wait for settings to be applied

    def return_to_start(self, keep_height: bool = False):
        """Safely brakes the drone and returns it to the given landing position.

        The drone is killed as soon as it tilts more than `MAX_TILT` on the way.

        Args:
            keep_height: Fly back at the current height, clipped to [0.5, 2.5] m, instead of
                climbing to the safe height of 2 m first.
        """
        # Enable high-level functions of the drone and disable low-level control access
        self.cf.param.set_value("stabilizer.controller", 2)
        self.cf.commander.send_stop_setpoint()
        self.cf.commander.send_notify_setpoint_stop()  # Tell drone to ignore low level setpoints
        self.cf.param.set_value("commander.enHighLevel", 1)
        self.cf.supervisor.send_arming_request(True)
        # Fly back to the start position
        BREAKING_DISTANCE = 1.0  # m
        BREAKING_DURATION = 1.5 if not keep_height else 0.75  # s
        RETURN_DURATION = 2.0  # s
        LAND_DURATION = 3.0  # s
        MAX_TILT = 90.0  # deg, tilted further the drone is killed
        MIN_BODY_Z = np.cos(np.deg2rad(MAX_TILT))  # vertical component of the z axis at MAX_TILT

        def killed_by_tilt() -> bool:
            body_z = R.from_quat(self._ros_connector.quat[self.name]).apply([0.0, 0.0, 1.0])
            if body_z[2] >= MIN_BODY_Z:
                return False
            self._emergency_stop()
            logger.warning(f"Return home failed: tilt above {MAX_TILT:.0f} deg, killed it")
            return True

        def wait_for_action(dt: float) -> bool:
            """Wait for the action to complete. Returns False if the drone was killed."""
            tstart = time.perf_counter()
            # Wait for the action to be completed and send the current position to the drone
            while time.perf_counter() - tstart < dt:
                if killed_by_tilt():
                    return False
                self.update_estimators()
                time.sleep(1 / self.pos_update_freq)
                if not self.healthy.is_set():
                    raise RuntimeError("Drone connection lost")
            return True

        pos = self._ros_connector.pos[self.name]
        vel = self._ros_connector.vel[self.name]
        vel_norm = np.linalg.norm(vel)
        break_pos = pos + vel / vel_norm * np.clip(vel_norm, 0.1, BREAKING_DISTANCE)
        # Elevate to a safe height to avoid obstacles while returning, or stay at the current one
        height = np.clip(pos[2], MIN_RETURN_HEIGHT, MAX_RETURN_HEIGHT) if keep_height else 2.0
        if keep_height:
            break_pos[2] = height
        self.cf.high_level_commander.go_to(*break_pos, 0, BREAKING_DURATION)
        if not wait_for_action(BREAKING_DURATION):
            return
        elevate_pos = break_pos.copy()
        elevate_pos[2] = height
        self.cf.high_level_commander.go_to(*elevate_pos, 0, BREAKING_DURATION)
        if not wait_for_action(BREAKING_DURATION):
            return
        return_pos = copy(self.pos_landing)
        return_pos[2] = height
        self.cf.high_level_commander.go_to(*return_pos, 0, RETURN_DURATION)
        if not wait_for_action(RETURN_DURATION):
            return
        return_pos[2] = self.pos_landing[2]
        self.cf.high_level_commander.go_to(*return_pos, 0, LAND_DURATION)
        wait_for_action(LAND_DURATION)

    def land(self, height: float = 0.1, duration: float = 3.0):
        """Safely lands the drone at its current position."""
        self.cf.commander.send_stop_setpoint()
        self.cf.commander.send_notify_setpoint_stop()  # Tell drone to ignore low level setpoints
        self.cf.param.set_value("commander.enHighLevel", 1)
        self.cf.supervisor.send_arming_request(True)
        self.cf.high_level_commander.land(height, duration)
        time.sleep(duration)
        self.cf.high_level_commander.stop()
