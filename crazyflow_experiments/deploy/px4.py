"""Interface to a PX4 vehicle through MAVROS.

The node runs on the companion computer next to the MAVROS instance that talks to the flight
controller. It streams setpoints into PX4's offboard mode and reads the onboard state estimate, so
the deployment scripts see the same observation dict and send the same actions as with the Crazyflie
interface in `deploy/crazyflie.py`.

Localization:
    "gps" anchors the global frame at the home position PX4 sets at arming, the takeoff position,
    and watches the RTK fix. "mocap" anchors it at the motion capture origin, and watches the motion
    capture pose on `vision_pose/pose`. Either measurement going stale outdates the observation,
    which the deployment scripts answer with an emergency landing.

Frames and units:
    The global frame is x north, y west and z up, MAVROS's is ENU, and this module converts at the
    ROS boundary. Quaternions are scalar-last, roll, pitch and yaw extrinsic xyz in rad with yaw
    zero facing north, positions in m, velocities in m/s, angular velocities in rad/s in the body
    frame (x forward, y left, z up) and the collective thrust in N.

Flight procedure:
    The pilot starts the script, then arms and flips the RC mode switch to OFFBOARD. From there the
    script flies: to the start pose, through the trajectory, into a hold. The pilot takes over by
    flipping the switch back, which PX4 obeys immediately whatever the script sends, and lands.
    Stale measurements end the flight in `land` instead, PX4's landing mode, which does not depend
    on them.

Requirements:
    - MAVROS runs on the companion computer (`ros2 launch mavros px4.launch`) with its gpsstatus
      plugin enabled. Attitude setpoints need the `thrust_scaling` parameter of its `setpoint_raw`
      plugin at 1.0, which this module sets itself.
    - `mavros_msgs` is built into `ros_ws` by `tools/setup_mavros_msgs.sh` on environment
      activation, pin `MAVROS_MSGS_REF` to the MAVROS version of the companion computer.
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import atexit
import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import rclpy
from crazyflow.control import Control
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TwistStamped, Vector3
from mavros_msgs.msg import GPSRAW, AttitudeTarget, HomePosition, PositionTarget, State
from mavros_msgs.srv import MessageInterval, SetMode
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray

from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.control.takeoff import TakeoffTrajectory

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OFFBOARD = State.MODE_PX4_OFFBOARD
LAND = State.MODE_PX4_LAND
# MAVROS's ENU frame (east, north, up) to the global frame (north, west, up) is a quarter turn about z
GLOBAL_FROM_ENU = R.from_euler("z", -np.pi / 2)
ENU_FROM_GLOBAL = GLOBAL_FROM_ENU.inv()
# MAVLink messages behind the observation by id, see https://mavlink.io/en/messages/common.html.
# MAVROS builds the local pose and velocity from LOCAL_POSITION_NED and the orientation and body
# rates from ATTITUDE_QUATERNION, or from ATTITUDE while no quaternion has arrived.
STREAMS = {"ATTITUDE": 30, "ATTITUDE_QUATERNION": 31, "LOCAL_POSITION_NED": 32}
# The receiver's solution, GPS_RAW_INT, arrives at the receiver rate at most and repeats otherwise.
GPS_STREAM = {"GPS_RAW_INT": 24}
GPS_RATE = 20.0  # Hz
UNKNOWN_UINT16, UNKNOWN_UINT8 = 2**16 - 1, 2**8 - 1  # MAVLink's markers for unknown values
RTK_FIX_TYPES = (GPSRAW.GPS_FIX_TYPE_RTK_FLOAT, GPSRAW.GPS_FIX_TYPE_RTK_FIXED)
# MAVROS scales the thrust of attitude setpoints by this factor and drops the setpoints while it is
# unset. `thrust2pwm` already produces PX4's normalized command in [0, 1], so no scaling.
THRUST_SCALING = 1.0
LOCALIZATIONS = ("gps", "mocap")  # Where the position estimate comes from, see the module docstring


class Drone:
    """Interface to a PX4 vehicle through MAVROS."""

    def __init__(
        self,
        thrust_curve: list[float],
        thrust_min: float,
        thrust_max: float,
        ctrl_interface: Control,
        namespace: str = "",
        name: str = "hb_x500",
        localization: str = "gps",
        stream_rate: float | None = 100.0,
        setpoint_freq: float = 50.0,
        timeout: float = 10.0,
        rtk_outdated_after: float = 1.0,
        vision_outdated_after: float = 0.2,
    ):
        """Connect to MAVROS, wait for the state estimate and start streaming hold setpoints.

        Args:
            thrust_curve: Coefficients [a, b, c] of one motor's thrust over the normalized PX4
                command, thrust = a pwm^2 + b pwm + c, see `sysid/fit_thrust_to_thrust.py`.
            thrust_min: Lowest thrust of one motor in N.
            thrust_max: Highest thrust of one motor in N.
            ctrl_interface: Control interface the actions are sent through.
            namespace: MAVROS namespace, e.g. "/drone1" if MAVROS was launched with one.
            name: Vehicle name, used in the `/drones/<name>/command` logging topic.
            localization: "gps" or "mocap", see the module docstring.
            stream_rate: Rate in Hz requested for the PX4 messages, None keeps the FCU's rates.
            setpoint_freq: Rate at which `_keepalive` repeats position setpoints.
            timeout: Seconds to wait for MAVROS.
            rtk_outdated_after: Seconds without an RTK fix before the observation is outdated.
            vision_outdated_after: Seconds without a motion capture pose before it is outdated.
        """
        if localization not in LOCALIZATIONS:
            raise ValueError(f"localization is one of {LOCALIZATIONS}, not {localization!r}")
        self.name = name
        self.ctrl_interface = ctrl_interface
        self._thrust_curve = tuple(thrust_curve)
        self._thrust_min, self._thrust_max = thrust_min, thrust_max
        self._setpoint_period = 1 / setpoint_freq
        self._mocap = localization == "mocap"
        self._outdated_after = vision_outdated_after if self._mocap else rtk_outdated_after

        self._lock = threading.Lock()
        # ENU position of the global origin. The estimator origin is the ENU origin itself, so
        # "mocap" needs no message and the frame is anchored from the start.
        self._origin: NDArray | None = np.zeros(3) if self._mocap else None
        self._pos: NDArray | None = None
        self._quat: NDArray | None = None
        self._vel: NDArray | None = None
        self._ang_vel: NDArray | None = None
        self._gps: dict[str, float | int] | None = None
        # Arrival of the measurements the estimate rests on, which the PX4 estimate cannot show: it
        # keeps streaming at its full rate while the estimator dead reckons on the IMU alone
        self._t_vision = -np.inf
        self._t_rtk = -np.inf
        self._mode, self._armed, self._connected = "", False, False
        self._target: PositionTarget | AttitudeTarget | None = None
        self._t_target = -np.inf
        self._counts = {"pose": 0, "vision": 0, "imu": 0, "state": 0, "gps": 0}

        self._owns_context = not rclpy.ok()
        if self._owns_context:
            # Ctrl+C has to reach the scripts as KeyboardInterrupt while the node is still alive, so
            # their cleanup runs. rclpy's own handlers would shut the context down first.
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node(f"px4_{name}")
        mavros = self._mavros = f"{namespace}/mavros"
        # MAVROS publishes and subscribes best effort, which a best effort subscription and a
        # reliable publisher are both compatible with.
        qos = qos_profile_sensor_data
        if self._mocap:
            # The motion capture pose on its way into PX4, published by the tracking node next to
            # MAVROS. Subscribing alongside MAVROS shows when the measurements stop arriving.
            self._node.create_subscription(
                PoseStamped, f"{mavros}/vision_pose/pose", self._on_vision, qos
            )
        else:
            self._node.create_subscription(
                HomePosition, f"{mavros}/home_position/home", self._on_home, qos
            )
        self._node.create_subscription(
            PoseStamped, f"{mavros}/local_position/pose", self._on_pose, qos
        )
        self._node.create_subscription(
            TwistStamped, f"{mavros}/local_position/velocity_local", self._on_velocity, qos
        )
        self._node.create_subscription(Imu, f"{mavros}/imu/data", self._on_imu, qos)
        self._node.create_subscription(State, f"{mavros}/state", self._on_state, qos)
        self._node.create_subscription(GPSRAW, f"{mavros}/gpsstatus/gps1/raw", self._on_gps, qos)
        self._position_pub = self._node.create_publisher(
            PositionTarget, f"{mavros}/setpoint_raw/local", 10
        )
        self._attitude_pub = self._node.create_publisher(
            AttitudeTarget, f"{mavros}/setpoint_raw/attitude", 10
        )
        self._cmd_pub = self._node.create_publisher(
            Float64MultiArray, f"/drones/{name}/command", 10
        )
        self._interval_client = self._node.create_client(
            MessageInterval, f"{mavros}/set_message_interval"
        )
        self._param_client = self._node.create_client(
            SetParameters, f"{mavros}/setpoint_raw/set_parameters"
        )
        self._mode_client = self._node.create_client(SetMode, f"{mavros}/set_mode")
        self._node.create_timer(self._setpoint_period, self._keepalive)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        self._closed = False
        atexit.register(self.close)  # A node left spinning aborts the interpreter at exit

        self._wait_for_data(timeout)
        if ctrl_interface == Control.attitude:
            self._set_thrust_scaling()
        if stream_rate is not None:
            self._request_stream_rates(stream_rate)
        self._log_stream_rates(stream_rate)
        origin = "motion capture" if self._mocap else "home"
        logger.info(
            f"Vehicle at {np.round(self.obs()['pos'], 2)} from the {origin} origin (north, west, up)"
        )

    @property
    def mode(self) -> str:
        """The PX4 flight mode, e.g. "AUTO.LOITER" for HOLD, "POSCTL" or "OFFBOARD"."""
        with self._lock:
            return self._mode

    @property
    def offboard(self) -> bool:
        """Whether PX4 is in OFFBOARD mode and follows the setpoints of this node."""
        return self.mode == OFFBOARD

    @property
    def armed(self) -> bool:
        """Whether the vehicle is armed."""
        with self._lock:
            return self._armed

    def obs(self) -> dict[str, NDArray]:
        """Generates an observation containing pos, quat, rpy, vel, ang_vel, is_outdated and gps."""
        with self._lock:
            obs = {
                "pos": self._pos.copy(),
                "quat": self._quat.copy(),
                "vel": self._vel.copy(),
                "ang_vel": self._ang_vel.copy(),
                "is_outdated": self._is_outdated(),
                "gps": None if self._gps is None else dict(self._gps),
            }
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        return obs

    def _is_outdated(self) -> bool:
        # The measurements the estimate rests on, not the estimate itself: the motion capture pose
        # with "mocap", the RTK fix of the receiver with "gps". Called under `_lock`.
        t_measurement = self._t_vision if self._mocap else self._t_rtk
        return time.monotonic() - t_measurement > self._outdated_after

    def _send_state(self, pos: NDArray, vel: NDArray, acc: NDArray, yaw: float):
        # PX4 tracks the position and takes the velocity and acceleration as feedforward
        self._publish_target(self._position_target(pos, vel, acc, yaw))
        self._publish_cmd(np.concatenate((pos, vel, acc, [yaw])))

    def send_action(self, action: NDArray):
        """Send an action through the control interface.

        Attitude actions are [roll, pitch, yaw, thrust] in rad and N, state actions are
        [pos, vel, acc, quat, body_rates] as in crazyflow, whose body rates PX4 cannot take.
        """
        match self.ctrl_interface:
            case Control.attitude:
                self._publish_target(self._attitude_target(action[:3], action[3]))
                self._publish_cmd(action)
            case Control.state:
                yaw = R.from_quat(action[9:13]).as_euler("xyz")[2]
                self._send_state(action[:3], action[3:6], action[6:9], yaw)
            case _:
                raise NotImplementedError(f"Unknown control interface {self.ctrl_interface}")

    def wait_for_offboard(self):
        """Block until the pilot switches to OFFBOARD with the RC mode switch."""
        if self.offboard:
            return
        armed = "armed" if self.armed else "not armed"
        logger.info(f"Vehicle in {self.mode}, {armed}. Flip the RC mode switch to OFFBOARD.")
        while rclpy.ok() and not self.offboard:
            time.sleep(0.05)

    def fly_to(
        self,
        target_pos: NDArray[np.floating],
        target_yaw: float,
        speed: float = 0.5,
        hover_time: float = 2.0,
        ctrl_freq: float = 50.0,
    ):
        """Fly a straight line at `speed` to the target pose and hover there for `hover_time`.

        Stops early if the pilot leaves OFFBOARD, and lands if the measurements are outdated.
        """
        obs = self.obs()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        ramp_time = max(float(np.linalg.norm(target_pos - obs["pos"])) / speed, 1.0)
        trajectory = TakeoffTrajectory(obs["pos"], obs["rpy"][2], target_pos, target_yaw, ramp_time)
        trajectory_pos, trajectory_yaw = trajectory.setpoints(ctrl_freq, hover_time)
        zeros = np.zeros(3)
        logger.info(f"Flying to {target_pos} in {ramp_time:.1f}s")
        for pos, yaw in zip(trajectory_pos, trajectory_yaw, strict=True):
            t = time.perf_counter()
            if not rclpy.ok() or not self.offboard:
                logger.warning("Left OFFBOARD, the pilot has control. Aborting the flight.")
                return
            if self.obs()["is_outdated"]:
                logger.warning("Measurements outdated. Aborting the flight.")
                self.land()
                return
            self._send_state(pos, zeros, zeros, yaw)
            sleep_until_next_cycle(t, ctrl_freq)

    def hold(self, brake_time: float = 0.5, brake_distance: float = 1.0):
        """Hold position `brake_time` ahead at the current velocity, at most `brake_distance` away."""
        obs = self.obs()
        brake = obs["vel"] * brake_time
        if (distance := np.linalg.norm(brake)) > brake_distance:
            brake *= brake_distance / distance
        self._send_state(obs["pos"] + brake, np.zeros(3), np.zeros(3), obs["rpy"][2])

    def land(self):
        """Hand the vehicle to PX4's landing mode, which descends without the lost measurements."""
        logger.warning(f"Landing: switching to {LAND}.")
        try:
            response = self._call(
                self._mode_client, SetMode.Request(custom_mode=LAND), "commanding a landing"
            )
        except RuntimeError:
            logger.exception(f"Could not reach MAVROS to switch to {LAND}. Take over on the RC.")
            return
        if not response.mode_sent:
            logger.error(f"MAVROS did not send {LAND}. Take over on the RC.")

    def wait_for_takeover(self):
        """Block until the pilot leaves OFFBOARD with the RC mode switch."""
        logger.info("Holding position. Flip the RC mode switch away from OFFBOARD to take over.")
        while rclpy.ok() and self.offboard:
            time.sleep(0.1)
        logger.info(f"The pilot has control, vehicle in {self.mode}.")

    def close(self):
        """Stop the setpoint stream and shut down the node. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._executor.shutdown()
        self._node.destroy_node()
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()

    def _spin(self):
        while not self._stop.is_set() and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.05)

    def _on_home(self, msg: HomePosition):
        # PX4 sets home at arming, so it is the takeoff position. MAVROS gives it in its ENU frame.
        home = np.array([msg.position.x, msg.position.y, msg.position.z])
        with self._lock:
            changed = self._origin is None or np.linalg.norm(home - self._origin) > 0.01
            self._origin = home
        if changed:
            logger.info(f"Home position, the global origin, at ENU {np.round(home, 2)}")

    def _on_pose(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        with self._lock:
            if self._origin is None:
                return  # Positions are relative to the origin of the global frame
            self._pos = GLOBAL_FROM_ENU.apply(np.array([p.x, p.y, p.z]) - self._origin)
            self._quat = (GLOBAL_FROM_ENU * R.from_quat([q.x, q.y, q.z, q.w])).as_quat()
            self._counts["pose"] += 1

    def _on_vision(self, msg: PoseStamped):
        del msg
        with self._lock:
            self._t_vision = time.monotonic()
            self._counts["vision"] += 1

    def _on_velocity(self, msg: TwistStamped):
        v = msg.twist.linear
        with self._lock:
            self._vel = GLOBAL_FROM_ENU.apply([v.x, v.y, v.z])

    def _on_imu(self, msg: Imu):
        w = msg.angular_velocity
        with self._lock:
            self._ang_vel = np.array([w.x, w.y, w.z])
            self._counts["imu"] += 1

    def _on_gps(self, msg: GPSRAW):
        def known(value: int, unknown: int, scale: float) -> float:
            return np.nan if value == unknown else value * scale

        gps = {
            "t": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            "fix_type": int(msg.fix_type),
            "satellites": known(msg.satellites_visible, UNKNOWN_UINT8, 1),
            "lat": msg.lat * 1e-7,
            "lon": msg.lon * 1e-7,
            "alt": msg.alt * 1e-3,
            "hdop": known(msg.eph, UNKNOWN_UINT16, 1e-2),
            "vdop": known(msg.epv, UNKNOWN_UINT16, 1e-2),
            "h_acc": msg.h_acc * 1e-3,
            "v_acc": msg.v_acc * 1e-3,
            "vel_acc": msg.vel_acc * 1e-3,
            "dgps_age": msg.dgps_age * 1e-3,
            "dgps_numch": int(msg.dgps_numch),
        }
        rtk = msg.fix_type in RTK_FIX_TYPES
        with self._lock:
            had_rtk = self._gps is not None and self._gps["fix_type"] in RTK_FIX_TYPES
            self._gps = gps
            if rtk:
                self._t_rtk = time.monotonic()
            self._counts["gps"] += 1
        if rtk != had_rtk:
            logger.info(f"GNSS fix type {gps['fix_type']}, RTK {'regained' if rtk else 'lost'}")

    def _on_state(self, msg: State):
        with self._lock:
            previous = self._mode
            self._mode, self._armed, self._connected = msg.mode, msg.armed, msg.connected
            self._counts["state"] += 1
        if msg.mode == previous:
            return
        logger.info(f"PX4 mode {previous or 'unknown'} -> {msg.mode}")
        if msg.mode == OFFBOARD:
            logger.info("OFFBOARD engaged, the script has control.")
        elif previous == OFFBOARD:
            logger.info("OFFBOARD left, the pilot has control.")

    def _keepalive(self):
        # PX4 gates OFFBOARD on a setpoint stream from both sides: it only accepts the switch once
        # setpoints have been arriving for a moment, and it drops back out through its failsafe once
        # they stop for `COM_OF_LOSS_T`. The control loop covers neither end, since it does not run
        # before the pilot switches over and pauses between phases, so this timer fills the gaps.
        # Outside of OFFBOARD it streams the current pose, so the switch never moves the vehicle. In
        # OFFBOARD it repeats the last position setpoint, which holds the vehicle between phases.
        # Attitude setpoints are never repeated: a stale attitude is flown open loop, so if the
        # control loop stalls it is better to let the stream lapse and the failsafe take over.
        with self._lock:
            offboard = self._mode == OFFBOARD
            target, t_target = self._target, self._t_target
            pos, quat = self._pos, self._quat
        if not offboard:
            if pos is not None:
                yaw = R.from_quat(quat).as_euler("xyz")[2]
                self._publish_target(self._position_target(pos, np.zeros(3), np.zeros(3), yaw))
        elif isinstance(target, PositionTarget):
            # Below a full period, or timer jitter would skip every other repetition
            if time.monotonic() - t_target >= 0.9 * self._setpoint_period:
                self._publish_target(target)

    def _publish_target(self, target: PositionTarget | AttitudeTarget):
        target.header.stamp = self._node.get_clock().now().to_msg()
        publisher = self._position_pub if isinstance(target, PositionTarget) else self._attitude_pub
        publisher.publish(target)
        with self._lock:
            self._target, self._t_target = target, time.monotonic()

    def _publish_cmd(self, cmd: NDArray):
        self._cmd_pub.publish(Float64MultiArray(data=np.asarray(cmd, dtype=np.float64)))

    def _position_target(
        self, pos: NDArray, vel: NDArray, acc: NDArray, yaw: float
    ) -> PositionTarget:
        """Build the setpoint in MAVROS's ENU frame, which MAVROS converts to PX4's NED."""
        target = PositionTarget()
        target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        target.type_mask = PositionTarget.IGNORE_YAW_RATE
        x, y, z = ENU_FROM_GLOBAL.apply(pos) + self._origin
        target.position = Point(x=x, y=y, z=z)
        x, y, z = ENU_FROM_GLOBAL.apply(vel)
        target.velocity = Vector3(x=x, y=y, z=z)
        x, y, z = ENU_FROM_GLOBAL.apply(acc)
        target.acceleration_or_force = Vector3(x=x, y=y, z=z)
        target.yaw = float(yaw) + np.pi / 2  # north is east plus a quarter turn
        return target

    def _attitude_target(self, rpy: NDArray, thrust: float) -> AttitudeTarget:
        """Build the setpoint as orientation of the body in ENU, MAVROS converts it for PX4."""
        target = AttitudeTarget()
        target.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE
            | AttitudeTarget.IGNORE_PITCH_RATE
            | AttitudeTarget.IGNORE_YAW_RATE
        )
        x, y, z, w = (ENU_FROM_GLOBAL * R.from_euler("xyz", rpy)).as_quat()
        target.orientation = Quaternion(x=x, y=y, z=z, w=w)
        target.thrust = thrust2pwm(thrust, self._thrust_curve, self._thrust_min, self._thrust_max)
        return target

    def _wait_for_data(self, timeout: float):
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                missing = [
                    name
                    for name, value in (
                        ("home position", self._origin),
                        ("local position", self._pos),
                        ("local velocity", self._vel),
                        ("IMU", self._ang_vel),
                    )
                    if value is None
                ]
                if self._mocap and self._t_vision == -np.inf:
                    missing.append("vision pose")
                if not self._mocap and self._t_rtk == -np.inf:
                    missing.append("RTK fix")
                if not self._connected:
                    missing.append("FCU connection")
            if not missing:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"No {', '.join(missing)} from MAVROS within {timeout}s. Is MAVROS running and "
                    f"connected to the flight controller?{self._missing_hint(missing)}"
                )
            time.sleep(0.05)

    def _missing_hint(self, missing: list[str]) -> str:
        if "vision pose" in missing:
            return (
                f" Nothing publishes {self._mavros}/vision_pose/pose, so the motion capture "
                "measurements cannot be watched for dropouts. Is the tracking node running and "
                "connected to the motion capture server?"
            )
        if "RTK fix" in missing:
            return (
                " The GNSS receiver has no RTK fix, or MAVROS does not publish GPS_RAW_INT. Is the "
                'RTK base station up and its correction stream running? Set localization to "mocap"'
                " to fly indoors on a motion capture estimate instead."
            )
        if "home position" in missing:
            return (
                " PX4 sets the home position at arming with a global position, which it never has "
                'indoors. Set localization to "mocap" to anchor the global frame at the estimator '
                "origin, which is the motion capture origin."
            )
        return ""

    def _set_thrust_scaling(self):
        """Set the thrust scaling of the MAVROS setpoint_raw plugin.

        The plugin drops every attitude setpoint while the parameter is unset.
        """
        value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=THRUST_SCALING)
        request = SetParameters.Request(parameters=[Parameter(name="thrust_scaling", value=value)])
        result = self._call(self._param_client, request, "setting the thrust scaling").results[0]
        if not result.successful:
            raise RuntimeError(f"MAVROS rejected thrust_scaling={THRUST_SCALING}: {result.reason}")
        logger.info(f"MAVROS thrust scaling set to {THRUST_SCALING}")

    def _request_stream_rates(self, rate: float):
        """Ask PX4 to send the messages behind the observation at the given rate."""
        # float, because the generated MessageInterval request aborts the interpreter on an int,
        # and the rate comes from a config where 100 parses as one
        rates = {name: float(rate) for name in STREAMS} | {name: GPS_RATE for name in GPS_STREAM}
        for name, message_id in (STREAMS | GPS_STREAM).items():
            request = MessageInterval.Request(message_id=message_id, message_rate=rates[name])
            what = (
                f"requesting {name} at {rates[name]} Hz. Pass stream_rate=None to keep the rates "
                "configured on the flight controller"
            )
            response = self._call(self._interval_client, request, what)
            if not response.success:
                raise RuntimeError(f"PX4 rejected {name} at {rates[name]} Hz")

    def _call(
        self, client: rclpy.client.Client, request: object, what: str, timeout: float = 5.0
    ) -> object:
        # Raises if MAVROS does not answer in time
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"The MAVROS service {client.srv_name} is not available")
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError(f"No answer from MAVROS within {timeout}s when {what}")
            time.sleep(0.01)
        return future.result()

    def _log_stream_rates(self, stream_rate: float | None, duration: float = 1.0):
        with self._lock:
            start = dict(self._counts)
        time.sleep(duration)
        with self._lock:
            rates = {k: (self._counts[k] - start[k]) / duration for k in start}
        vision = f"vision pose {rates['vision']:.0f} Hz, " if self._mocap else ""
        logger.info(
            f"MAVROS streams: local position {rates['pose']:.0f} Hz, {vision}"
            f"attitude {rates['imu']:.0f} Hz, state {rates['state']:.1f} Hz, "
            f"GPS {rates['gps']:.1f} Hz"
        )
        if stream_rate is not None and rates["pose"] < 0.8 * stream_rate:
            logger.warning(f"Local position arrives below the requested {stream_rate} Hz")
        if not self._mocap and rates["gps"] < 2 / self._outdated_after:
            logger.warning(
                f"GPS arrives at {rates['gps']:.1f} Hz, too slow to watch the RTK fix over "
                f"{self._outdated_after * 1e3:.0f} ms. The flight will abort on a false alarm."
            )


def thrust2pwm(
    thrust: float, thrust_curve: tuple[float, float, float], thrust_min: float, thrust_max: float
) -> float:
    """Invert the thrust curve thrust = a pwm^2 + b pwm + c of one motor, for a collective thrust.

    The curve rises over the command range and the thrust limits lie inside the range it covers, so
    clipping the per motor thrust to them first keeps the root real. The root that continues the
    linear term is the one inside the command range, for either sign of a.
    """
    a, b, c = thrust_curve
    thrust_per_motor = np.clip(thrust / 4, thrust_min, thrust_max)
    if a == 0.0:  # A linear curve, e.g. the Crazyflie's, has no quadratic term to divide by
        pwm = (thrust_per_motor - c) / b
    else:
        pwm = (-b + np.sqrt(b**2 - 4 * a * (c - thrust_per_motor))) / (2 * a)
    return float(np.clip(pwm, 0.0, 1.0))
